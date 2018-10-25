import json
from collections import Counter
from typing import Dict, Optional

import numpy as np
import torch
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from overrides import overrides
from tabulate import tabulate

# TODO: Explore stopless version.
from library.dataset_readers.util import STOP_WORDS
from library.models.vae import VAE
from torch.nn.functional import log_softmax


@Model.register("BOWTopicModel")
class BOWTopicModel(Model):
    """
    Topic model with VAE, training the VAE in a semi-supervised
    environment (https://arxiv.org/abs/1406.5298).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
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
                 vae: VAE,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 background_data_path: str = None,
                 update_bg: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BOWTopicModel, self).__init__(vocab, regularizer)

        self.metrics = {
            'KL-Divergence': Average(),
            'Reconstruction': Average()
        }

        self.vocab = vocab
        self.vae = vae

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
        vocab_size = self.vocab.get_vocab_size("stopless")
        self.batchnorm = torch.nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.batchnorm.weight.data.copy_(torch.from_numpy(np.ones(vocab_size)))
        self.batchnorm.weight.requires_grad = False

        # Learnable bias and latent topics.
        #bg_freq_file = '../student/data/tam/bgfreq.json'
        #alpha = torch.FloatTensor(self.vocab.get_vocab_size("stopless"))
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

        beta = torch.FloatTensor(self.latent_dim, self.vocab.get_vocab_size("stopless"))
        self.beta = torch.nn.Parameter(beta)
        torch.nn.init.uniform_(self.beta)

        # For computing metrics.
        self.kl_divergence = 0
        self.reconstruction = 0

        self.step = 0

        initializer(self)

    @overrides
    def forward(self,
                input_tokens: torch.Tensor,
                sentiment: torch.Tensor,
                labelled: torch.Tensor):
        """
        Computes loss for unlabelled data.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)
        :param args: misc.
            We only need the input tokens, this allows flexibility in the dataset reader used.

        Returns ELBO loss.
        """
        output_dict = {}
        self.kl_divergence = 0
        self.reconstruction = 0

        elbo = self.ELBO(input_tokens)

        output_dict['loss'] = -torch.sum(elbo)
        self.metrics['KL-Divergence'](self.kl_divergence)
        self.metrics['Reconstruction'](self.reconstruction)

        if self.step == 100:
            print(tabulate(self.extract_topics(), headers=["Topic #", "Words"]))
            self.step = 0
        else:
            if self.training:
                self.step += 1

        return output_dict

    def ELBO(self, input_tokens: torch.Tensor):  # pylint: disable=C0103
        """
        Computes ELBO loss: -KL-Divergence + reconstruction log likelihood.

        :param input_tokens: ``torch.Tensor``
            The tokenized input, expected as (batch, sequence length)

        Returns both ELBO and the classification logits for convenience.
        """

        # Bag-of-words representation.
        # bow = compute_bow_vector(self.vocab, input_tokens)
        bow = self._compute_stopless_bow(input_tokens)

        # Variational Inference.
        z, mu, sigma = self.vae(bow)  # pylint: disable=C0103

        z_do = self.z_dropout(z)

        # For better interpretibility of topics.
        theta = torch.softmax(z_do, dim=-1)  # pylint: disable=C0103

        # Reconstruction log likelihood.
        reconstructed_bow = theta.matmul(self.beta) + self.alpha

        reconstructed_bow_bn = self.batchnorm(reconstructed_bow)

        log_reconstructed_bow = log_softmax(reconstructed_bow_bn + 1e-10, dim=-1)
        reconstruction_log_likelihood = torch.sum(bow * log_reconstructed_bow, dim=-1)

        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum(dim=-1)
        self.kl_divergence = -torch.mean(negative_kl_divergence).item()
        self.reconstruction = -torch.mean(reconstruction_log_likelihood)

        # ELBO = - KL-Div(q(z | x, y), P(z)) +  E[P(x | z, y)]
        # Shape: (batch, )
        elbo = reconstruction_log_likelihood + negative_kl_divergence

        return elbo

    def _compute_stopless_bow(self, input_tokens: Dict[str, torch.Tensor]):
        """
        Return a vector representation of words in the stopless dimension.

        :param input_tokens: ``torch.Tensor``
            A (batch, sequence length) size tensor.
        """

        batch_size = input_tokens['tokens'].size(0)
        stopless_bow = torch.zeros(batch_size, self.vocab.get_vocab_size("stopless")).float()
        stopless_token_to_index = self.vocab.get_token_to_index_vocabulary('stopless')
        for row, example in enumerate(input_tokens['tokens']):
            word_counts = Counter()
            words = [self.vocab.get_token_from_index(index.item()) for index in example]
            word_counts.update(words)

            # Remove padding and unknown tokens.
            words = filter(lambda x: x not in (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN), words)
            for word in words:
                if word in stopless_token_to_index:
                    # Increment the count at the word's stopless index for the current row.
                    stopless_word_index = self.vocab.get_token_index(word, "stopless")
                    stopless_bow[row][stopless_word_index] = word_counts[word]

        return stopless_bow


    def _compute_background_log_frequency(self, precomputed_word_counts: str):
        """ Load in the word counts from the JSON file and compute the
            background log term frequency w.r.t this vocabulary. """
        precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
        log_term_frequency = torch.FloatTensor(self.vocab.get_vocab_size())
        for i in range(self.vocab.get_vocab_size()):
            token = self.vocab.get_token_from_index(i)
            if token == DEFAULT_OOV_TOKEN or token == DEFAULT_PADDING_TOKEN:
                log_term_frequency[i] = 1e-12
            elif token in precomputed_word_counts:
                log_term_frequency[i] = precomputed_word_counts[token]

        log_term_frequency = torch.log(log_term_frequency)

        # At this point, padding and UNKNOWN are -inf.
        #log_term_frequency[0] = 0
        #log_term_frequency[1] = 0

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
