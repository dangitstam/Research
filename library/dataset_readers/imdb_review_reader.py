import logging
from typing import Dict

from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides

from .util import normalize_raw_text, STOP_WORDS

import ujson

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("imdb_review_reader")
class IMDBReviewReader(DatasetReader):
    """
    Reads the 100K IMDB dataset in format that it appears in
    http://ai.stanford.edu/~amaas/data/sentiment/
    (i.e. this reader expects a full-path to the directory as a result of
     extracting the tar).

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split text into English tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    words_per_instance : ``int``, optional
        The number of words in which the raw text will be bucketed to allow for more efficient
        training (backpropagation-through-time limit).
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer(namespace="tokens", lowercase_tokens=True),
            "stopless_tokens": SingleIdTokenIndexer(namespace="stopless_tokens", lowercase_tokens=True)
        }


    @overrides
    def _read(self, file_path):
        # A training instance consists of the word frequencies for the entire review and a
        # `words_per_instance`` portion of the review.
        file_path = cached_path(file_path)
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                example = ujson.loads(line)
                example_text = normalize_raw_text(example['text'])
                example_text_tokenized = self._tokenizer.tokenize(example_text)
                example_sentiment = 1 if example['sentiment'] >= 5 else 0
                exampled_labelled = 1 if example['sentiment'] else 0
                example_instance = {
                    'input_tokens': TextField(example_text_tokenized, self._token_indexers),
                    'sentiment': LabelField(example_sentiment, skip_indexing=True),
                    'labelled': LabelField(exampled_labelled, skip_indexing=True)
                }

                yield Instance(example_instance)


@DatasetReader.register("processed_document_reader")
class DocumentReader(DatasetReader):
    """
    Assumes that documents have been preprocessed into a .jsonl such that each
    object contains the full text, the filtered text, and class label.

    Each ``read`` yields a data instance of
        input_tokens: The full text, altered only by adding UNK tokens and padding.
        filtered_tokens: The full text with stopwords omitted.
        sentiment: Either `positive` or `negative`.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split text into English tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define English token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer(namespace="en", lowercase_tokens=True)}``.
    words_per_instance : ``int``, optional
        The number of words in which the raw text will be bucketed to allow for more efficient
        training (backpropagation-through-time limit).
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._stopless_token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace="stopless", lowercase_tokens=True)
        }
        self._token_indexers = {
            "tokens": SingleIdTokenIndexer(namespace="full", lowercase_tokens=True)
        }


    @overrides
    def _read(self, file_path):
        # A training instance consists of the word frequencies for the entire review and a
        # `words_per_instance`` portion of the review.
        file_path = cached_path(file_path)
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                example = ujson.loads(line)
                example_text = example['text']
                example_filtered_text = example['stopless']
                example_text_tokenized = self._tokenizer.tokenize(example_text)
                example_filtered_text_tokenized = self._tokenizer.tokenize(example_filtered_text)
                example_sentiment = 1 if example['sentiment'] >= 5 else 0
                exampled_labelled = 1 if example['sentiment'] else 0
                example_instance = {
                    'input_tokens': TextField(example_text_tokenized, self._token_indexers),
                    'filtered_tokens': TextField(example_filtered_text_tokenized, self._stopless_token_indexers),
                    'sentiment': LabelField(example_sentiment, skip_indexing=True),
                    'labelled': LabelField(exampled_labelled, skip_indexing=True)
                }

                yield Instance(example_instance)
