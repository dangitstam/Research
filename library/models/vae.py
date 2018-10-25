from typing import Dict

import numpy as np
import torch
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from overrides import overrides


@Model.register("vae")
class VAE(Model):
    def __init__(self, vocab,
                 encoder: FeedForward,
                 mu_projection: FeedForward,
                 log_variance_projection: FeedForward,
                 decoder: FeedForward,
                 apply_batchnorm: Dict[str, bool] = None):
        super(VAE, self).__init__(vocab)
        assert mu_projection.get_output_dim() == log_variance_projection.get_output_dim() and \
               mu_projection.get_output_dim() == log_variance_projection.get_output_dim(), \
            "Gaussian MLP's input and output sizes must match."

        assert encoder.get_output_dim() == mu_projection.get_input_dim(), \
            "Encoder's output dim {} differs from the Gaussian MLP's input sizes {} for mu and log variance." \
                .format(encoder.get_output_dim(), mu_projection.get_input_dim())

        assert decoder.get_input_dim() == mu_projection.get_output_dim(), \
            "Decoder's input dim {} differs from the Gaussian MLP's output sizes {} for mu and log variance." \
                .format(decoder.get_input_dim(), mu_projection.get_output_dim())

        self.encoder = encoder
        self.mu_projection = mu_projection
        self.log_variance_projection = log_variance_projection
        self.decoder = decoder
        self.dropout = torch.nn.Dropout(0.2)

        self.batchnorm = torch.nn.BatchNorm1d(mu_projection.get_output_dim(), eps=0.001, momentum=0.001, affine=True)
        self.batchnorm.weight.data.copy_(torch.from_numpy(np.ones(mu_projection.get_output_dim())))
        self.batchnorm.weight.requires_grad = False

        self.apply_batchnorm = apply_batchnorm

    @overrides
    def forward(self, input_vector: torch.Tensor):  # pylint: disable=W0221
        """
        Given an input vector, produces the latent encoding z, followed by the mean and
        log variance of the variational distribution produced.

        z is the result of the reparameterization trick (Autoencoding Variational Bayes (Kingma et al.)).
        """
        initial_latent_projection = self.encoder(input_vector)
        initial_latent_projection_do = self.dropout(initial_latent_projection)
        mu = self.mu_projection(initial_latent_projection_do)  # pylint: disable=C0103

        if self.apply_batchnorm['mu']:
            mu = self.batchnorm(mu) # pylint: disable=C0103

        log_variance = self.log_variance_projection(initial_latent_projection)

        # Generate random noise.
        # Shape: (batch, latent_dim)
        epsilon = torch.randn(mu.size(0), mu.size(-1))

        # Extract sigma and reparameterize.
        sigma = torch.sqrt(torch.exp(log_variance))
        if self.apply_batchnorm['sigma']:
            sigma = self.batchnorm(sigma) # pylint: disable=C0103

        if self.training:
            z = mu + sigma * epsilon # pylint: disable=C0103
        else:
            z = mu

        return z, mu, sigma
