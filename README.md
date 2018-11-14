# Exploration in Utilizing In-Domain, Unlabelled Data in Text Classification

## Getting Started

### Requirements

Install all dependencies using
```
pip install -r requirements.txt
```

The code base largely relies on [AllenNLP](https://github.com/allenai/allennlp).

### Generating the preprocessed data
Currently, there is full support for IMDB, which can be obtained [here](http://ai.stanford.edu/~amaas/data/sentiment/). Once obtained, unzip the file and pass the path to the resulting `aclImdb` directory to `configure_data.sh` along with the desired destination to save the processed datafiles:

```
bash configure_data.sh <path to aclImdb/> <path to desired destination directory>
```

If the destination directory already exists, running this script will not overwrite existing files but will instead add and append to files of the same names.

### Running Experiments

Experiments are run via AllenNLP's config system. To train the bag-of-embeddings baseline, run

```
allennlp train experiments/supervised/boe.json -s saved_models/boe --include-package library
```
