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
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear

from library.dataset_readers.util import STOP_WORDS

from .util import compute_stopless_bow_vector


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
                input_tokens: Dict[str, torch.LongTensor],
                sentiment: Dict[str, torch.LongTensor]):

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
        sigma = self.sigma_projection(init_latent_bow)

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
