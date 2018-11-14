import json
from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn as nn
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides
from tabulate import tabulate
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import log_softmax
from torch.nn.modules.linear import Linear

from library.dataset_readers.util import STOP_WORDS
from library.models.vae import VAE

from .util import (compute_stopless_bow_vector, log_standard_categorical,
                   separate_labelled_and_unlabelled_instances)


@Model.register("seq2vec_classifier")
class Seq2VecClassifier(Model):
    """ Simple generalized sequence encodding to classification.
    """
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2VecClassifier, self).__init__(vocab, regularizer)

        self.metrics = {
            'accuracy': CategoricalAccuracy()
        }

        self.encoder = encoder
        self.output_projection = Linear(self.encoder.get_output_dim(), 2)
        self.text_field_embedder = text_field_embedder
        self.criterion = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                input_tokens: Dict[str, torch.LongTensor],
                sentiment: Dict[str, torch.LongTensor]):

        output_dict = {}

        # Encode the current input text, incorporating previous hidden state if available.
        # Shape: (batch x BPTT limit x hidden size)
        embedded_input = self.text_field_embedder(input_tokens)
        input_mask = util.get_text_field_mask(input_tokens)
        encoded_input = self.encoder(embedded_input, input_mask)

        # Compute logits.
        # Shape: (batch x num_classes)
        logits = self.output_projection(encoded_input)

        # Loss and metrics.
        output_dict['loss'] = self.criterion(logits, sentiment)
        self.metrics['accuracy'](logits, sentiment)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


@Model.register("seq2vec_vae_classifier")
class Seq2VecClassifierVAE(Model):
    """ Generalized sequence encodding to classification supplemented with a VAE
        that reconstructs the bag-of-words represnetation.
    """
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 text_field_embedder: TextFieldEmbedder,
                 latent_dim: int = 512,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2VecClassifierVAE, self).__init__(vocab, regularizer)

        self.metrics = {
            'KL-Divergence': Average(),
            'Accuracy': CategoricalAccuracy()
        }

        self.vocab = vocab
        self.encoder = encoder
        self.text_field_embedder = text_field_embedder
        self.latent_dim = latent_dim
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.reconstruction_criterion = torch.nn.MSELoss()

        # Produce the namespace that excludes stopwords.
        self.stopless_namespace = "stopless"
        for word, _ in self.vocab.get_token_to_index_vocabulary().items():
            if word not in STOP_WORDS:
                self.vocab.add_token_to_namespace(word, self.stopless_namespace)
        self.stopless_dim = self.vocab.get_vocab_size(self.stopless_namespace)

        feature_dim = self.encoder.get_output_dim() + self.latent_dim
        self.output_projection = Linear(feature_dim, 2)

        # The VAE first projections word-frequencies via a feedforward network g(.).
        # Mean mu is computed via W_mu(g(.)) + a_mu.
        # Sigma is computed as W_sigma(g(.)) + a_sigma.
        self.initial_latent_projection = FeedForward(
            self.stopless_dim,
            2,
            [self.stopless_dim // 2, latent_dim * 2],
            [torch.nn.ReLU(), torch.nn.ReLU()]
        )
        self.mu_projection = FeedForward(
            latent_dim * 2,
            2,
            [latent_dim * 2, latent_dim],
            [torch.nn.ReLU(), torch.nn.ReLU()]
        )
        self.sigma_projection = FeedForward(
            latent_dim * 2,
            2,
            [latent_dim * 2, latent_dim],
            [torch.nn.ReLU(), torch.nn.ReLU()]
        )

        # The latent topics learned.
        self.beta = torch.FloatTensor(self.latent_dim, self.vocab.get_vocab_size(self.stopless_namespace))
        self.beta = torch.nn.Parameter(self.beta)
        torch.nn.init.uniform_(self.beta)

        # Noise used to implement the reparameterization trick.
        # Prior will be a 0, 1 multivariate gaussian.
        self.noise = MultivariateNormal(torch.zeros(latent_dim), torch.eye(self.latent_dim))

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                input_tokens: Dict[str, torch.Tensor],
                sentiment: torch.Tensor,
                labelled: torch.Tensor):

        output_dict = {}

        # Encode the current input text, incorporating previous hidden state if available.
        # Shape: (batch x BPTT limit x hidden size)
        embedded_input = self.text_field_embedder(input_tokens)
        input_mask = util.get_text_field_mask(input_tokens)
        encoded_input = self.encoder(embedded_input, input_mask)

        # Bag-of-words representation.
        # TODO: init latent proj needs to know the stopless dim...
        stopless_bow = compute_stopless_bow_vector(self.vocab, input_tokens, self.stopless_namespace)

        # Compute the variational distribution.
        init_latent_bow = self.initial_latent_projection(stopless_bow)
        mu = self.mu_projection(init_latent_bow)  # pylint: disable=C0103
        log_sigma_squared = self.sigma_projection(init_latent_bow)
        sigma = torch.sqrt(torch.exp(log_sigma_squared))

        # Sample from the VAE.
        epsilon = self.noise.rsample(sample_shape=torch.Size([mu.size(0)])).to(mu.device)
        latent_bow = mu + sigma * epsilon

        features = torch.cat([encoded_input, latent_bow], dim=-1)

        # Train the VAE to make the latent features rich.
        reconstructed_bow = latent_bow.matmul(self.beta)   # TODO: Background freq.
        reconstruction_loss = self.reconstruction_criterion(reconstructed_bow, stopless_bow)

        # Compute logits.
        # Shape: (batch x num_classes)
        logits = self.output_projection(features)
        classification_loss = self.classification_criterion(logits, sentiment)

        # Loss and metrics.
        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum()

        # Joint learning of classification and the VAE.
        # Negative KL-div needs to be negated since loss is negative likelihood.
        output_dict['loss'] = -negative_kl_divergence + reconstruction_loss + classification_loss
        self.metrics['KL-Divergence'](-negative_kl_divergence.item())
        self.metrics['Accuracy'](logits, sentiment)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


########################################################
####### Baseline Semi-Supervised BoW model (original)
##############


@Model.register("BOWTopicModelSemiSupervisedBaseline")
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
                 classification_layer: FeedForward,
                 vae: VAE,
                 alpha: float = 0.1,
                 background_data_path: str = None,
                 update_bg: bool = True,
                 shared_representation: bool = False,
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
        self.classification_layer = classification_layer
        self.vae = vae
        self.alpha = alpha
        self.num_classes = classification_layer.get_output_dim()
        self.shared_representation = shared_representation

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

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

        # Compute supervised and unsupervised objective.
        L, classification_loss = self.ELBO(labelled_instances)  # pylint: disable=C0103
        U = self.U(unlabelled_instances)   # pylint: disable=C0103

        labelled_loss = -torch.sum(L)
        # When evaluating, there is no unlabelled data.
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

    def ELBO(self, instances: Dict, sentiment: Optional[torch.Tensor] = None):  # pylint: disable=C0103
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

        # Regardless of label, every instance has a full and filtered version.
        ids = instances['ids']
        input_tokens = instances['tokens']
        filtered_tokens = instances['stopless_tokens']

        # Labelled instances will already have sentiment.
        if instances['labelled']:
            sentiment = instances['sentiment']

        batch_size = input_tokens.size(0)

        # Bag-of-words representation.
        bow = self._compute_stopless_bow(ids, filtered_tokens).to(device=self.device)

        # One-hot the sentiment vector before concatenation.
        sentiment_one_hot = torch.FloatTensor(batch_size, self.num_classes).to(device=self.device)
        sentiment_one_hot.zero_()
        sentiment_one_hot = sentiment_one_hot.scatter_(1, sentiment.reshape(-1, 1), 1)

        # Variational Inference, where Z ~ q(z | x, y)
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

        # For convenience: allows proper mapping from example to sentiment if
        # classification loss is optionally returned here.
        if instances['labelled']:
            logits = self.classification_layer(bow)
            classification_loss = self.classification_criterion(logits, sentiment)

            # Update metrics.
            self.metrics['Accuracy'](logits, sentiment)
            self.metrics['Cross_Entropy'](classification_loss)

            return elbo, classification_loss

        return elbo

    def U(self, instances: Dict):  # pylint: disable=C0103
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """

        # Regardless of label, every instance has a full and filtered version.
        ids = instances['ids']
        input_tokens = instances['tokens']
        filtered_tokens = instances['stopless_tokens']

        batch_size = input_tokens.size(0)

        # No work to be done on zero instances.
        if batch_size == 0:
            return None

        elbos = torch.zeros(self.num_classes, batch_size).to(device=self.device)
        for i in range(self.num_classes):
            # Instantiate an artifical labelling for each class.
            # Labels are treated as a latent variable that we marginalize over.
            sentiment = (torch.ones(batch_size).long() * i).to(device=self.device)
            elbos[i] = self.ELBO(instances, sentiment=sentiment)

        # Bag-of-words representation.
        bow = self._compute_stopless_bow(ids, filtered_tokens).to(device=self.device)

        # Compute q(y | x).
        # Shape: (batch, num_classes)
        logits = torch.softmax(self.classification_layer(bow), dim=-1)

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
