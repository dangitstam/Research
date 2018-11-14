import json
from collections import Counter
from typing import Dict, Optional

from tabulate import tabulate
import torch
from torch.nn.functional import log_softmax

from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)

from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides

from library.models.vae import VAE

from .util import log_standard_categorical, separate_labelled_and_unlabelled_instances


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
                 encoder: Seq2VecEncoder,
                 classification_layer: FeedForward,
                 vae: VAE,
                 input_embedder: TextFieldEmbedder,
                 filtered_embedder: TextFieldEmbedder,
                 alpha: float = 0.1,
                 background_data_path: str = None,
                 update_bg: bool = True,
                 use_filtered_tokens: bool = True,
                 use_shared_representation: bool = False,
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

        self.input_embedder = input_embedder
        self.filtered_embedder = filtered_embedder
        self.encoder = encoder
        self.vocab = vocab
        self.classification_layer = classification_layer
        self.vae = vae
        self.alpha = alpha
        self.num_classes = classification_layer.get_output_dim()

        # Hyperparameter flags.
        self._use_shared_representation = use_shared_representation
        self._use_filtered_tokens = use_filtered_tokens

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # Note that the VAE's encoder is the initial projection into the latent space.
        self.latent_dim = vae.mu_projection.get_output_dim()

        # Batchnorm to be applied throughout inference.
        stopless_vocab_size = self.vocab.get_vocab_size("stopless")
        self.batchnorm = torch.nn.BatchNorm1d(stopless_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.batchnorm.weight.data.copy_(torch.ones(stopless_vocab_size, dtype=torch.float64))
        self.batchnorm.weight.requires_grad = False

        # Input batchnorm when the representation is shared.
        self.input_batchnorm = torch.nn.BatchNorm1d(self.encoder.get_output_dim(), eps=0.001, momentum=0.001, affine=True)
        self.input_batchnorm.weight.data.copy_(torch.ones(self.encoder.get_output_dim(), dtype=torch.float64))
        self.input_batchnorm.weight.requires_grad = False

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

        # Cache bows for faster training.
        self._id_to_bow = {}

        # For easy tranfer to the GPU.
        self.device = beta.device

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                ids: torch.Tensor,
                input_tokens: Dict[str, torch.LongTensor],
                filtered_tokens: Dict[str, torch.LongTensor],
                sentiment: torch.Tensor,
                labelled: torch.Tensor):

        output_dict = {}

        # Sort instances into labelled and unlabelled portions.
        labelled_instances, unlabelled_instances = separate_labelled_and_unlabelled_instances(
            ids, input_tokens['tokens'], filtered_tokens['tokens'], sentiment, labelled)

        # Stopless Bag-of-Words to be reconstructed.
        labelled_bow = self._compute_stopless_bow(labelled_instances['ids'],
                                                  labelled_instances['stopless_tokens'])

        # Logits for labelled data.
        labelled_logits, labelled_encoded_input = self._classify(labelled_instances)
        labelled_logits = self.classification_layer(labelled_encoded_input)

        labelled_sentiment = labelled_instances['sentiment']

        # Compute supervised and unsupervised objective.
        L = self.ELBO(labelled_encoded_input, labelled_bow, labelled_sentiment)

        # # When provided, use the unlabelled data.
        if len(unlabelled_instances['ids']) > 0:
            unlabelled_bow = self._compute_stopless_bow(unlabelled_instances['ids'],
                                                        unlabelled_instances['stopless_tokens'])

            # Logits for unlabelled data where the label is a latent variable.
            unlabelled_logits, unlabelled_encoded_input = self._classify(unlabelled_instances)

            # Logits need to be normalized for proper weighting.
            unlabelled_logits = torch.softmax(unlabelled_logits, dim=-1)

            U = self.U(unlabelled_encoded_input, unlabelled_bow, unlabelled_logits)
        else:
            U = None

        if self.training:
            assert U is not None, "Current batch does not contain unlabelled data."

        # Classification loss and metrics.
        classification_loss = self.classification_criterion(
            labelled_logits, labelled_sentiment)
        self.metrics['Accuracy'](labelled_logits, labelled_sentiment)
        self.metrics['Cross_Entropy'](classification_loss)

        labelled_loss = -torch.sum(L)
        unlabelled_loss = -torch.sum(U if U is not None else torch.FloatTensor([0]))
        self.metrics['ELBO']((labelled_loss + unlabelled_loss).item())

        # Joint supervised and unsupervised learning.
        scaled_classification_loss = self.alpha * classification_loss
        J_alpha = (labelled_loss + unlabelled_loss) + scaled_classification_loss  # pylint: disable=C0103

        output_dict['loss'] = J_alpha

        # While training, it's helpful to see how the topics are changing.
        if self.training and self.step == 100:
            print(tabulate(self.extract_topics(self.beta), headers=["Topic #", "Words"]))
            print(tabulate(self.extract_topics(self.covariates), headers=["Covariate #", "Words"]))
            self.step = 0
        else:
            if self.training:
                self.step += 1

        return output_dict

    def _classify(self, instances: Dict):
        """
        Produce logits here
        """
        if self._use_filtered_tokens:
            embedded_tokens = self.filtered_embedder({"filtered": instances['stopless_tokens']})
            encoded_input = self.encoder(embedded_tokens)
            labelled_logits = self.classification_layer(encoded_input)
        else:
            embedded_tokens = self.filtered_embedder({"full": instances['tokens']})
            encoded_input = self.encoder(embedded_tokens)
            labelled_logits = self.classification_layer(encoded_input)

        return labelled_logits, encoded_input

    def ELBO(self,  # pylint: disable=C0103
             input_representation: torch.Tensor,
             bow: torch.Tensor,
             sentiment: torch.Tensor):
        """
        Computes ELBO loss. For convenience, also returns classification loss
        given the label.

        :param instances: ``Dict``
            Instances that contain either labelled or unlabelled data, but not both.
            If the instances are labelled, `sentiment` is an expected key that
            should contain the sentiment for each example.
        :param sentiment: ``torch.Tensor``
            The target class labels, expexted as (batch,). Used only for
            computing the unlabelled objective; this sentiment is treated
            as a latent variable unlike the sentiment provided in labelled
            versions of `instances`.
        """
        batch_size = input_representation.size(0)

        # One-hot the sentiment vector before concatenation.
        sentiment_one_hot = torch.FloatTensor(batch_size, self.num_classes).to(device=self.device)
        sentiment_one_hot.zero_()
        sentiment_one_hot = sentiment_one_hot.scatter_(1, sentiment.reshape(-1, 1), 1)

        # Variational Inference, where Z ~ q(z | x, y) OR Z ~ q(z | h(x), y)
        if self._use_shared_representation:
            input_representation = self.input_batchnorm(input_representation)
            z, mu, sigma = self.vae(torch.cat((input_representation, sentiment_one_hot), dim=-1))  # pylint: disable=C0103
        else:
            z, mu, sigma = self.vae(torch.cat((bow, sentiment_one_hot), dim=-1))  # pylint: disable=C0103

        # For better interpretibility of topics.
        theta = torch.softmax(z, dim=-1)  # pylint: disable=C0103

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

        return elbo

    def U(self,  # pylint: disable=C0103
          input_representation: torch.Tensor,
          bow: torch.Tensor,
          logits: torch.Tensor):
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """
        batch_size = input_representation.size(0)

        # No work to be done on zero instances.
        if batch_size == 0:
            return None

        elbos = torch.zeros(self.num_classes, batch_size).to(device=self.device)
        for i in range(self.num_classes):
            # Instantiate an artifical labelling for each class.
            # Labels are treated as a latent variable that we marginalize over.
            sentiment = (torch.ones(batch_size).long() * i).to(device=self.device)
            elbos[i] = self.ELBO(input_representation, bow, sentiment)

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        # Reshape elbos to be (batch, num_classes) before the per-class weighting.
        L_weighted = torch.sum(logits * elbos.t(), dim=-1)  # pylint: disable=C0103
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1)  # pylint: disable=C0103

        return L_weighted + H

    def _compute_stopless_bow(self,
                              ids: torch.Tensor,
                              input_tokens: Dict[str, torch.Tensor]):
        """
        Return a vector representation of words in the stopless dimension.

        :param input_tokens: ``torch.Tensor``
            A (batch, sequence length) size tensor.
        """

        batch_size = input_tokens.size(0)
        stopless_bow = torch.zeros(batch_size, self.vocab.get_vocab_size("stopless")).float()
        for row, example in enumerate(input_tokens):
            example_id = ids[row].item()
            if example_id in self._id_to_bow:
                stopless_bow[row] = self._id_to_bow[example_id].clone()
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

            self._id_to_bow[example_id] = stopless_bow[row].clone()

        return stopless_bow

    def _compute_background_log_frequency(self, precomputed_word_counts: str):
        """
        Load in the word counts from the JSON file and compute the
        background log term frequency w.r.t this vocabulary.

        :param precomputed_word_counts: ``str``
            The path to the JSON object containing word to frequency mappings.
        """
        precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
        log_term_frequency = torch.FloatTensor(self.vocab.get_vocab_size('stopless'))
        for i in range(self.vocab.get_vocab_size('stopless')):
            token = self.vocab.get_token_from_index(i, 'stopless')
            if token in (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN):
                log_term_frequency[i] = 1e-12
            elif token in precomputed_word_counts:
                log_term_frequency[i] = precomputed_word_counts[token]

        log_term_frequency = torch.log(log_term_frequency)

        return log_term_frequency

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}

    def extract_topics(self, weights: torch.Tensor, k: int = 20):
        """
        Given the learned (K, vocabulary size) weights, print the
        top k words from each row as a topic.

        :param k: ``int``
            The number of words per topic to display.
        """

        print()  # Start on a new line for readibility.
        words = list(range(weights.size(1)))
        words = [self.vocab.get_token_from_index(i, "stopless") for i in words]

        topics = []

        word_strengths = list(zip(words, self.background.tolist()))
        sorted_by_strength = sorted(word_strengths,
                                    key=lambda x: x[1],
                                    reverse=True)
        background = [x[0] for x in sorted_by_strength][:k]
        topics.append(('bg', background))

        for i, topic in enumerate(weights):
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            topic = [x[0] for x in sorted_by_strength][:k]
            topics.append((i, topic))

        return topics
