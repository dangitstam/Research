import argparse
import html
import os
import random
import re
import sys
from collections import Counter

from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from tqdm import tqdm

import ujson


def main():
    """
    Given a corpora, produces a dictionary of word counts, saving the
    result as a JSON file.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--data-path", type=str,
                        help="Path to the IMDB training file.")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the word counts.")
    parser.add_argument("--seed", type=int,
                        default=1337,
                        help="Random seed to use when shuffling data.")
    args = parser.parse_args()

    tokenizer = WordTokenizer()
    word_counts = Counter()
    print("Collecting word counts:")
    with open(args.data_path, 'r') as data_file:
        for line in tqdm(data_file):
            line = line.strip("\n")
            if not line:
                continue
            example = ujson.loads(line)
            example_text = normalize_raw_text(example['text'])

            # Must collect explicit strings, otherwise the counter will accumulate
            # counts of references.
            example_text_tokenized = [token.string.strip() for token in tokenizer.tokenize(example_text)]
            word_counts.update(example_text_tokenized)

    ujson.dump(word_counts, open(os.path.join(args.save_dir, "word_counts.json"), "w"))

def normalize_raw_text(text):
    """ Copied from dataset_readers.util to prevent hacks.
    """
    re1 = re.compile(r'  +')
    text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ','.').replace(
                ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(text)).lower()

if __name__ == "__main__":
    main()
