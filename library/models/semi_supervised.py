import json
from typing import Dict, Optional

import torch
from torch.nn.modules.linear import Linear

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides

from library.dataset_readers.util import STOP_WORDS  # TODO: Explore stopless version.
from library.models.vae import VAE

from .util import (compute_bow_vector, log_standard_categorical,
                   sort_unsupervised_instances)


@Model.register("BOWSeq2VecClassifier")
class BOWSeq2VecClassifier(Model):
    """
    Joint topic model and text classifier, training the VAE in a semi-supervised
    environment (https://arxiv.org/abs/1406.5298).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    encoder : ``Seq2VecEncoder``
        The encoder used to encode input text.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    vae : ``VAE``
        The variational autoencoder used to project the BoW into a latent space.
    precompued_word_counts: ``str``
        Path to a JSON file containing word counts accumulated over the training corpus.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2VecEncoder,
                 vae: VAE,
                 text_field_embedder: TextFieldEmbedder,
                 precomputed_word_counts: str = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BOWSeq2VecClassifier, self).__init__(vocab, regularizer)

        self.metrics = {
            'KL-Divergence': Average(),
            'Accuracy': CategoricalAccuracy()
        }

        self.vocab = vocab
        self.encoder = encoder
        self.vae = vae
        self.text_field_embedder = text_field_embedder

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss()
        self.reconstruction_criterion = torch.nn.MSELoss()

        # Background log frequency vector.
        self.log_term_frequency = self._compute_background_log_frequency(precomputed_word_counts)

        # Note that the VAE's encoder is the initial projection into the latent space.
        self.latent_dim = vae.encoder.get_output_dim()
        feature_dim = self.encoder.get_output_dim() + self.latent_dim
        self.output_projection = Linear(feature_dim, 2)

        # The latent topics learned.
        self.beta = torch.FloatTensor(self.latent_dim, self.vocab.get_vocab_size())
        self.beta = torch.nn.Parameter(self.beta)
        torch.nn.init.uniform_(self.beta)

        # Given there's two loss functions that both have a KL-Divergence term, to better keep track of it
        # as a metric, store it here and zero it out before computing loss.
        self.kl_divergence = 0

        initializer(self)

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                input_tokens: Dict[str, torch.LongTensor],
                sentiment: torch.Tensor,
                labelled: torch.Tensor):

        output_dict = {}
        self.kl_divergence = 0

        (labelled_tokens, labelled_sentiment), (unlabelled_tokens, _) = \
            sort_unsupervised_instances(input_tokens['tokens'], sentiment, labelled)

        L, classification_loss = self.supervised_forward(labelled_tokens, labelled_sentiment)  # pylint: disable=C0103
        U = self.unsupervised_forward(unlabelled_tokens)   # pylint: disable=C0103

        # Joint supervised and unsupervised learning. 'classification_loss' is already cross entropy loss
        # so there's no need to negate it.
        output_dict['loss'] = -L + classification_loss - U
        self.metrics['KL-Divergence'](self.kl_divergence)

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

        elbo, logits = self.ELBO(input_tokens)

        # TODO: Alpha parameter on classification loss.
        # Supplementary classification loss.
        classification_loss = self.classification_criterion(logits, sentiment)
        self.metrics['Accuracy'](logits, sentiment)

        return torch.mean(elbo), torch.mean(classification_loss)

    def unsupervised_forward(self, input_tokens: torch.Tensor): # pylint: disable=C0103
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns the sum of ELBO and the entropy of the predicted classification logits.
        """
        if input_tokens.size(0) == 0:
            return 0

        elbo, logits = self.ELBO(input_tokens)

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1) # pylint: disable=C0103
        L = torch.sum(logits * elbo, dim=-1) # pylint: disable=C0103

        return torch.mean(L + H)

    def ELBO(self, input_tokens: torch.Tensor):  # pylint: disable=C0103
        """
        Computes ELBO loss:
        KL-Divergence, reconstruction, and log likelihood of the labels assuming a uniform prior.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns both ELBO and the classification logits for convenience.
        """

        # Encode the current input text, incorporating previous hidden state if available.
        # Shape: (batch x BPTT limit x hidden size)
        input_tokens = {'tokens': input_tokens}  # AllenNLP constructs expect a dictionary.
        embedded_input = self.text_field_embedder(input_tokens)
        input_mask = util.get_text_field_mask(input_tokens)
        encoded_input = self.encoder(embedded_input, input_mask)

        # Bag-of-words representation.
        bow = compute_bow_vector(self.vocab, input_tokens)

        # Variational Inference.
        z, mu, sigma = self.vae(bow)  # pylint: disable=C0103

        # For better interpretibility of topics.
        z = torch.softmax(z, dim=-1)

        features = torch.cat([encoded_input, z], dim=-1)

        # Train the VAE to make the latent features rich.
        reconstructed_bow = z.matmul(self.beta) + self.log_term_frequency
        reconstruction_loss = self.reconstruction_criterion(reconstructed_bow, bow)

        # Compute logits.
        # Shape: (batch x num_classes)
        logits = self.output_projection(features)

        # We assume a uniform prior for y.
        label_prior = -log_standard_categorical(logits)

        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum()
        self.kl_divergence += -negative_kl_divergence.item()

        # Joint learning of classification and the VAE for labeled instances.
        # ELBO = E[P(x | z, y) + P(y)] - KL-Div(q(z | x, y), P(z))
        elbo = -reconstruction_loss + label_prior + negative_kl_divergence

        return elbo, logits

    def _compute_background_log_frequency(self, precomputed_word_counts: str):
        """ Load in the word counts from the JSON file and compute the
            background log term frequency w.r.t this vocabulary. """
        precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
        log_term_frequency = torch.FloatTensor(self.vocab.get_vocab_size())
        for i in range(self.vocab.get_vocab_size()):
            token = self.vocab.get_token_from_index(i)
            if token in precomputed_word_counts:
                self.log_term_frequency[i] = precomputed_word_counts[token]
        log_term_frequency = torch.log(self.log_term_frequency / torch.sum(log_term_frequency))

        return log_term_frequency

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
