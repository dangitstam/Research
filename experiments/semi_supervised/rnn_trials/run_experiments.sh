#!/bin/sh

allennlp train experiments/semi_supervised/rnn_trials/rnn0.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn0
rm -rf experiments/semi_supervised/rnn_trials/results/rnn0/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn0/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn0/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn1.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn1
rm -rf experiments/semi_supervised/rnn_trials/results/rnn1/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn1/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn1/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn2.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn2
rm -rf experiments/semi_supervised/rnn_trials/results/rnn2/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn2/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn2/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn3.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn3
rm -rf experiments/semi_supervised/rnn_trials/results/rnn3/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn3/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn3/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn4.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn4
rm -rf experiments/semi_supervised/rnn_trials/results/rnn4/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn4/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn4/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn5.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn5
rm -rf experiments/semi_supervised/rnn_trials/results/rnn5/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn5/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn5/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn6.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn6
rm -rf experiments/semi_supervised/rnn_trials/results/rnn6/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn6/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn6/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn7.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn7
rm -rf experiments/semi_supervised/rnn_trials/results/rnn7/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn7/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn7/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn8.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn8
rm -rf experiments/semi_supervised/rnn_trials/results/rnn8/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn8/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn8/test.txt
allennlp train experiments/semi_supervised/rnn_trials/rnn9.json --include-package library -s experiments/semi_supervised/rnn_trials/results/rnn9
rm -rf experiments/semi_supervised/rnn_trials/results/rnn9/*.th
allennlp evaluate experiments/semi_supervised/rnn_trials/results/rnn9/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/rnn_trials/results/rnn9/test.txt
