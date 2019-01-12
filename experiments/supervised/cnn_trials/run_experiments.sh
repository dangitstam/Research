#!/bin/sh

allennlp train experiments/supervised/cnn_trials/cnn0.json --include-package library -s experiments/supervised/cnn_trials/results/cnn0
rm -rf experiments/supervised/cnn_trials/results/cnn0/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn0/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn0/test.txt
allennlp train experiments/supervised/cnn_trials/cnn1.json --include-package library -s experiments/supervised/cnn_trials/results/cnn1
rm -rf experiments/supervised/cnn_trials/results/cnn1/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn1/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn1/test.txt
allennlp train experiments/supervised/cnn_trials/cnn2.json --include-package library -s experiments/supervised/cnn_trials/results/cnn2
rm -rf experiments/supervised/cnn_trials/results/cnn2/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn2/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn2/test.txt
allennlp train experiments/supervised/cnn_trials/cnn3.json --include-package library -s experiments/supervised/cnn_trials/results/cnn3
rm -rf experiments/supervised/cnn_trials/results/cnn3/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn3/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn3/test.txt
allennlp train experiments/supervised/cnn_trials/cnn4.json --include-package library -s experiments/supervised/cnn_trials/results/cnn4
rm -rf experiments/supervised/cnn_trials/results/cnn4/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn4/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn4/test.txt
allennlp train experiments/supervised/cnn_trials/cnn5.json --include-package library -s experiments/supervised/cnn_trials/results/cnn5
rm -rf experiments/supervised/cnn_trials/results/cnn5/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn5/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn5/test.txt
allennlp train experiments/supervised/cnn_trials/cnn6.json --include-package library -s experiments/supervised/cnn_trials/results/cnn6
rm -rf experiments/supervised/cnn_trials/results/cnn6/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn6/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn6/test.txt
allennlp train experiments/supervised/cnn_trials/cnn7.json --include-package library -s experiments/supervised/cnn_trials/results/cnn7
rm -rf experiments/supervised/cnn_trials/results/cnn7/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn7/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn7/test.txt
allennlp train experiments/supervised/cnn_trials/cnn8.json --include-package library -s experiments/supervised/cnn_trials/results/cnn8
rm -rf experiments/supervised/cnn_trials/results/cnn8/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn8/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn8/test.txt
allennlp train experiments/supervised/cnn_trials/cnn9.json --include-package library -s experiments/supervised/cnn_trials/results/cnn9
rm -rf experiments/supervised/cnn_trials/results/cnn9/*.th
allennlp evaluate experiments/supervised/cnn_trials/results/cnn9/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/supervised/cnn_trials/results/cnn9/test.txt
