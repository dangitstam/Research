import json
from collections import Counter
from typing import Dict, Optional

from tabulate import tabulate
import torch
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from torch.nn.functional import log_softmax
from torch.nn.modules.linear import Linear

from library.models.vae import VAE

from .util import (compute_bow_vector, log_standard_categorical,
                   sort_unsupervised_instances)

# TODO: A seq2vec addition will require a separate dataset reader.
# Construct a new dataset that contains sentiment / full text / filtered text.

@Model.register("BOWTopicModelSemiSupervised")
class BOWTopicModelSemiSupervised(Model):
    """
    VAE topic model trained in a semi-supervised environment
    (https://arxiv.org/abs/1406.5298).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    classification_layer : ``Feedfoward``
        Projection from latent topics to classification logits.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    vae : ``VAE``
        The variational autoencoder used to project the BoW into a latent space.
    background_data_path: ``str``
        Path to a JSON file containing word frequencies accumulated over the training corpus.
    update_bg: ``bool``:
        Whether to allow the background frequency to be learnable.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 vae: VAE,
                 classification_layer: FeedForward,
                 background_data_path: str = None,
                 update_bg: bool = True,
                 alpha: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BOWTopicModelSemiSupervised, self).__init__(vocab, regularizer)

        self.metrics = {
            'KL-Divergence': Average(),
            'Reconstruction': Average(),
            'Accuracy': CategoricalAccuracy(),
            'Cross_Entropy': Average(),
            'ELBO': Average()
        }

        self.vocab = vocab
        self.vae = vae
        self.classification_layer = classification_layer
        self.num_classes = classification_layer.get_output_dim()
        self.alpha = alpha

        self.z_dropout = torch.nn.Dropout(p=0.2)

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss()

        # Note that the VAE's encoder is the initial projection into the latent space.
        self.latent_dim = vae.mu_projection.get_output_dim()

        # Batchnorm to be applied throughout inference.
        stopless_vocab_size = self.vocab.get_vocab_size("stopless")
        self.batchnorm = torch.nn.BatchNorm1d(stopless_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.batchnorm.weight.data.copy_(torch.ones(stopless_vocab_size, dtype=torch.float64))
        self.batchnorm.weight.requires_grad = False

        # Learnable bias.
        if background_data_path is not None:
            background = self._compute_background_log_frequency(background_data_path)
            if update_bg:
                self.background = torch.nn.Parameter(background, requires_grad=True)
            else:
                self.background = torch.nn.Parameter(background, requires_grad=False)
        else:
            background = torch.FloatTensor(self.vocab.get_vocab_size("stopless"))
            self.background = torch.nn.Parameter(background)
            torch.nn.init.uniform_(self.background)

        # Learnable latent topics.
        beta = torch.FloatTensor(self.latent_dim, self.vocab.get_vocab_size("stopless"))
        self.beta = torch.nn.Parameter(beta)
        torch.nn.init.uniform_(self.beta)

        # Learnable covariates to relate latent topics and labels.
        covariates = torch.FloatTensor(self.num_classes, self.vocab.get_vocab_size("stopless"))
        self.covariates = torch.nn.Parameter(covariates)
        torch.nn.init.uniform_(self.covariates)

        # For computing metrics and printing topics.
        self.step = 0

        # Bow cache: computing bag-of-words everytime is expensive.
        self._id_to_bow = {}

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                id: torch.LongTensor,
                input_tokens: Dict[str, torch.LongTensor],
                filtered_tokens: Dict[str, torch.LongTensor],
                sentiment: torch.Tensor,
                labelled: torch.Tensor):

        output_dict = {}

        (labelled_id, _, labelled_filtered_tokens, labelled_sentiment), (unlabelled_id, _, unlabelled_filtered_tokens) = \
            sort_unsupervised_instances(id, input_tokens['tokens'], filtered_tokens['tokens'], sentiment, labelled)

        L, classification_loss = self.ELBO(labelled_id, labelled_filtered_tokens, labelled_sentiment)  # pylint: disable=C0103
        U = self.U(unlabelled_id, unlabelled_filtered_tokens)   # pylint: disable=C0103

        # Joint supervised and unsupervised learning. 
        # classification_loss' is already cross entropy loss so there's no need to negate it.
        labelled_loss = -torch.sum(L)

        # Note that in evaluation, there is no unlabelled data.
        unlabelled_loss = -torch.sum(U if U is not None else torch.FloatTensor([0]))
        self.metrics['ELBO']((labelled_loss + unlabelled_loss).item())

        # Classification loss is significantly smaller than the elbo objective; scale it so that
        # enough gradient flows through for learning.
        equalizer = ((labelled_loss + unlabelled_loss) / classification_loss).item()

        J_alpha = (labelled_loss + unlabelled_loss) + (self.alpha * equalizer) * classification_loss  # pylint: disable=C0103

        output_dict['loss'] = J_alpha

        # While training, it's helpful to see how the topics are changing.
        if self.training and self.step == 100:
            print(tabulate(self.extract_topics(), headers=["Topic #", "Words"]))
            self.step = 0
        else:
            if self.training:
                self.step += 1

        return output_dict

    def ELBO(self,
             id: torch.Tensor,
             input_tokens: torch.Tensor,  # pylint: disable=C0103
             sentiment: torch.Tensor,
             true_labelled: bool = True):
        """
        Computes ELBO loss for labelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)
        :param sentiment: ``torch.Tensor``
            The target class labels, expexted as (batch,)

        """
        device = input_tokens.device
        batch_size = input_tokens.size(0)

        # Bag-of-words representation.
        bow = self._compute_stopless_bow(id, input_tokens).to(device=device)

        # One-hot the sentiment vector before concatenation.
        sentiment_one_hot = torch.FloatTensor(batch_size, self.num_classes).to(device=device)
        sentiment_one_hot.zero_()
        sentiment_one_hot = sentiment_one_hot.scatter_(1, sentiment.reshape(-1, 1), 1)

        # Variational Inference, where Z ~ q(z | x, y)
        z, mu, sigma = self.vae(torch.cat((bow, sentiment_one_hot), dim=-1))  # pylint: disable=C0103
        z_do = self.z_dropout(z)

        # For better interpretibility of topics.
        theta = torch.softmax(z_do, dim=-1)  # pylint: disable=C0103

        reconstruction_bow = self.background + theta.matmul(self.beta) + self.covariates[sentiment]
        reconstructed_bow_bn = self.batchnorm(reconstruction_bow)

        # log P(x | y, z) = log softmax(b + z beta + y C)
        # Final shape: (batch, )
        log_reconstruction_bow = log_softmax(reconstructed_bow_bn + 1e-10, dim=-1)
        reconstruction_log_likelihood = torch.sum(bow * log_reconstruction_bow, dim=-1)

        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum(dim=-1)  # Shape: (batch, )

        # Uniform prior.
        prior = -log_standard_categorical(sentiment_one_hot)

        # ELBO = - KL-Div(q(z | x, y), P(z)) +  E[ log P(x | z, y) + log p(y) ]
        elbo = negative_kl_divergence + reconstruction_log_likelihood + prior

        # Update metrics
        self.metrics['KL-Divergence'](-torch.mean(negative_kl_divergence))
        self.metrics['Reconstruction'](-torch.mean(reconstruction_log_likelihood))

        # For convenience: allows proper mapping from example to sentiment if
        # classification loss is optionally returned here.
        if true_labelled:
            logits = self.classification_layer(bow)
            classification_loss = self.classification_criterion(logits, sentiment)

            # Update metrics.
            self.metrics['Accuracy'](logits, sentiment)
            self.metrics['Cross_Entropy'](classification_loss)

            return elbo, classification_loss

        return elbo

    def U(self, id: torch.Tensor, input_tokens: torch.Tensor): # pylint: disable=C0103
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """
        device = input_tokens.device
        batch_size = input_tokens.size(0)
        if batch_size == 0:
            return None

        elbos = torch.zeros(self.num_classes, batch_size).to(device=device)
        for i in range(self.num_classes):
            # Instantiate an artifical labelling for each class.
            # Labels are treated as a latent variable that we marginalize over.
            sentiment = (torch.ones(batch_size).long() * i).to(device=device)
            elbos[i] = self.ELBO(id, input_tokens, sentiment, true_labelled=False)

        # Bag-of-words representation.
        bow = self._compute_stopless_bow(id, input_tokens).to(device=device)

        # Compute q(y | x).
        # Shape: (batch, num_classes)
        logits = torch.softmax(self.classification_layer(bow), dim=-1)

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        # Reshape elbos to be (batch, num_classes) before the per-class weighting.
        L_weighted = torch.sum(logits * elbos.t(), dim=-1) # pylint: disable=C0103
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1) # pylint: disable=C0103

        return L_weighted + H

    def _compute_stopless_bow(self, id: torch.Tensor, input_tokens: torch.Tensor):
        """
        Return a vector representation of words in the stopless dimension.

        :param input_tokens: ``torch.Tensor``
            A (batch, sequence length) size tensor.
        """

        batch_size = input_tokens.size(0)
        stopless_bow = torch.zeros(batch_size, self.vocab.get_vocab_size("stopless")).float()
        for row, example in enumerate(input_tokens):
            if id[row] in self._id_to_bow:
                stopless_bow[row] = self._id_to_bow[id[row]].clone()
                continue

            word_counts = Counter()
            words = [self.vocab.get_token_from_index(index.item(), 'stopless') for index in example]
            word_counts.update(words)

            # Remove padding and unknown tokens.
            words = filter(lambda x: x not in (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN), words)
            for word in words:
                # Increment the count at the word's stopless index for the current row.
                stopless_word_index = self.vocab.get_token_index(word, "stopless")
                stopless_bow[row][stopless_word_index] = word_counts[word]

            self._id_to_bow[row] = stopless_bow[row].clone()

        return stopless_bow

    def _compute_background_log_frequency(self, precomputed_word_counts: str):
        """ Load in the word counts from the JSON file and compute the
            background log term frequency w.r.t this vocabulary. """
        precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
        log_term_frequency = torch.FloatTensor(self.vocab.get_vocab_size('stopless'))
        for i in range(self.vocab.get_vocab_size('stopless')):
            token = self.vocab.get_token_from_index(i, 'stopless')
            if token == DEFAULT_OOV_TOKEN or token == DEFAULT_PADDING_TOKEN:
                log_term_frequency[i] = 1e-12
            elif token in precomputed_word_counts:
                log_term_frequency[i] = precomputed_word_counts[token]

        log_term_frequency = torch.log(log_term_frequency)

        return log_term_frequency

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}

    def extract_topics(self, k=20):
        """
        Given the learned (topic, vocabulary size) beta, print the
        top k words from each topic.
        """

        words = list(range(self.beta.size(1)))
        words = [self.vocab.get_token_from_index(i, "stopless") for i in words]

        topics = []

        word_strengths = list(zip(words, self.background.tolist()))
        sorted_by_strength = sorted(word_strengths,
                                    key=lambda x: x[1],
                                    reverse=True)
        background = [x[0] for x in sorted_by_strength][:k]
        topics.append(('bg', background))

        for i, topic in enumerate(self.beta):
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            topic = [x[0] for x in sorted_by_strength][:k]
            topics.append((i, topic))

        return topics
