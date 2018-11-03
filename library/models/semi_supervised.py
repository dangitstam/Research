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
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BOWTopicModelSemiSupervised, self).__init__(vocab, regularizer)

        self.metrics = {
            'KL-Divergence': Average(),
            'Reconstruction': Average(),
            'Accuracy': CategoricalAccuracy(),
            'Cross_Entropy': Average()
        }

        self.vocab = vocab
        self.vae = vae
        self.classification_layer = classification_layer
        self.num_classes = classification_layer.get_output_dim()

        self.z_dropout = torch.nn.Dropout(p=0.2)

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss()

        # Note that the VAE's encoder is the initial projection into the latent space.
        self.latent_dim = vae.mu_projection.get_output_dim()

        # Establish the stopless dimension.
        for _, word in self.vocab.get_index_to_token_vocabulary().items():
            #if word not in STOP_WORDS:
            self.vocab.add_token_to_namespace(word, "stopless")

        # Batchnorm to be applied throughout inference.
        stopless_vocab_size = self.vocab.get_vocab_size("stopless")
        self.batchnorm = torch.nn.BatchNorm1d(stopless_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.batchnorm.weight.data.copy_(torch.ones(stopless_vocab_size, dtype=torch.float64))
        self.batchnorm.weight.requires_grad = False

        # Learnable bias.
        if background_data_path is not None:
            alpha = self._compute_background_log_frequency(background_data_path)
            if update_bg:
                self.alpha = torch.nn.Parameter(alpha, requires_grad=True)
            else:
                self.alpha = torch.nn.Parameter(alpha, requires_grad=False)
        else:
            alpha = torch.FloatTensor(self.vocab.get_vocab_size("stopless"))
            self.alpha = torch.nn.Parameter(alpha)
            torch.nn.init.uniform_(self.alpha)

        # Learnable latent topics.
        beta = torch.FloatTensor(self.latent_dim, self.vocab.get_vocab_size("stopless"))
        self.beta = torch.nn.Parameter(beta)
        torch.nn.init.uniform_(self.beta)

        # Learnable covariates to relate latent topics and labels.
        covariates = torch.FloatTensor(self.num_classes, self.vocab.get_vocab_size("stopless"))
        self.covariates = torch.nn.Parameter(covariates)
        torch.nn.init.uniform_(self.covariates)

        # For computing metrics and printing topics.
        self.kl_divergence = 0
        self.reconstruction = 0
        self.step = 0

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                input_tokens: Dict[str, torch.LongTensor],
                filtered_tokens: Dict[str, torch.LongTensor],
                sentiment: torch.Tensor,
                labelled: torch.Tensor):

        output_dict = {}
        self.kl_divergence = 0

        (_, labelled_filtered_tokens, labelled_sentiment), (_, unlabelled_filtered_tokens) = \
            sort_unsupervised_instances(input_tokens['tokens'], filtered_tokens['tokens'], sentiment, labelled)

        L, classification_loss = self.supervised_forward(labelled_filtered_tokens, labelled_sentiment)  # pylint: disable=C0103
        U = self.unsupervised_forward(unlabelled_filtered_tokens)   # pylint: disable=C0103

        # Joint supervised and unsupervised learning. 'classification_loss' is already cross entropy loss
        # so there's no need to negate it.
        output_dict['loss'] = torch.sum(-L + classification_loss - U, dim=0)
        self.metrics['KL-Divergence'](self.kl_divergence)
        self.metrics['Reconstruction'](self.reconstruction)

        # While training, it's helpful to see how the topics are changing.
        if self.training and self.step == 100:
            print(tabulate(self.extract_topics(), headers=["Topic #", "Words"]))
            self.step = 0
        else:
            if self.training:
                self.step += 1

        return output_dict

    def supervised_forward(self,  # pylint: disable=C0103
                           input_tokens: torch.Tensor,
                           sentiment: torch.Tensor):
        """
        Computes loss for labelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)
        :param sentiment: ``torch.Tensor``
            The target class labels, expexted as (batch,)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """
        if input_tokens.size(0) == 0:
            return 0, 0

        elbo, logits = self.ELBO({'tokens': input_tokens})

        # Collect the correct loss since the class labels are known.
        elbo = torch.gather(elbo, 1, sentiment.reshape(-1, 1))

        # TODO: Alpha parameter on classification loss.
        # Supplementary classification loss.
        classification_loss = self.classification_criterion(logits, sentiment)

        # Scale classification loss by 0.1 * N (it is much smaller than elbo).
        classification_loss *= 0.1 * (45000 / 20000)

        self.metrics['Accuracy'](logits, sentiment)
        self.metrics['Cross_Entropy'](classification_loss)

        return torch.sum(elbo), torch.sum(classification_loss)

    def unsupervised_forward(self, input_tokens: torch.Tensor): # pylint: disable=C0103
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """
        if input_tokens.size(0) == 0:
            return 0

        elbo, logits = self.ELBO({'tokens': input_tokens})

        # Normalize logits as they would be before prediction.
        logits = torch.softmax(logits, dim=-1)

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1) # pylint: disable=C0103
        L = torch.sum(logits * elbo, dim=-1) # pylint: disable=C0103

        return L + H


    def ELBO(self, input_tokens: torch.Tensor):  # pylint: disable=C0103
        """
        Computes ELBO loss: -KL-Divergence + reconstruction log likelihood.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns both ELBO and the classification logits for convenience.
        """

        device = input_tokens['tokens'].device

        # Bag-of-words representation.
        bow = self._compute_stopless_bow(input_tokens).to(device=device)

        # Variational Inference.
        z, mu, sigma = self.vae(bow)  # pylint: disable=C0103

        z_do = self.z_dropout(z)

        # For better interpretibility of topics.
        theta = torch.softmax(z_do, dim=-1)  # pylint: disable=C0103

        # Compute q_phi(y | x).
        logits = self.classification_layer(bow)

        # reconstruction log likelihood.
        # TODO: add y here (two ways?)
        # TODO: Return a (classes, batch, vocabulary) tensor;
        # - Supervised loss will take the tensor of the correct class for loss.
        # - Unsupervised loss wil interpolate between the two to marginalize on y.
        reconstruct_bow = theta.matmul(self.beta) + self.alpha

        batch_size = theta.size(0)
        stopless_vocab_size = self.vocab.get_vocab_size("stopless")

        # Expand covariates to (batch, classes, vocabulary).
        expanded_covariates = self.covariates.expand(batch_size, self.num_classes, stopless_vocab_size)

        # Temporarily reshape to (class, batch, vocabulary) and allow broadcasting to
        # properly add the bow (batch, vocabulary)
        reconstruct_bow_cov = expanded_covariates.reshape(self.num_classes, batch_size, -1) + reconstruct_bow
        reconstruct_bow_cov = reconstruct_bow_cov.reshape(batch_size, self.num_classes, -1)

        # Batchnorm expects the input size to always be the second dimension.
        reconstruct_bow_cov_bn = self.batchnorm(reconstruct_bow_cov
                                                .reshape(-1, stopless_vocab_size, self.num_classes))
        reconstruct_bow_cov_bn = reconstruct_bow_cov_bn.reshape(-1, self.num_classes, stopless_vocab_size)

        # Log loss on empirical frequency.
        log_reconstruct_bow = log_softmax(reconstruct_bow_cov_bn + 1e-10, dim=-1)
        probabilities = (log_reconstruct_bow.reshape(-1, batch_size, stopless_vocab_size) * bow)

        # Final shape: (batch, classes)
        reconstruction_log_likelihood = torch.sum(probabilities, dim=-1).reshape(batch_size, -1)

        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum(dim=-1)  # Shape: (batch, )
        self.kl_divergence = -torch.mean(negative_kl_divergence).item()
        self.reconstruction = -torch.mean(reconstruction_log_likelihood)

        # ELBO = - KL-Div(q(z | x, y), P(z)) +  E[ log P(x | z, y) ]
        # The reshape is necessary since we are now adding the KL-divergence across classes.
        elbo = negative_kl_divergence.reshape(-1, 1) + reconstruction_log_likelihood

        return elbo, logits

    def _compute_stopless_bow(self, input_tokens: Dict[str, torch.Tensor]):
        """
        Return a vector representation of words in the stopless dimension.

        :param input_tokens: ``torch.Tensor``
            A (batch, sequence length) size tensor.
        """

        batch_size = input_tokens['tokens'].size(0)
        stopless_bow = torch.zeros(batch_size, self.vocab.get_vocab_size("stopless")).float()
        for row, example in enumerate(input_tokens['tokens']):
            word_counts = Counter()
            words = [self.vocab.get_token_from_index(index.item(), 'stopless') for index in example]
            word_counts.update(words)

            # Remove padding and unknown tokens.
            words = filter(lambda x: x not in (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN), words)
            for word in words:
                # Increment the count at the word's stopless index for the current row.
                stopless_word_index = self.vocab.get_token_index(word, "stopless")
                stopless_bow[row][stopless_word_index] = word_counts[word]

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

        word_strengths = list(zip(words, self.alpha.tolist()))
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
