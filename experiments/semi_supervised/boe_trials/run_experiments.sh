#!/bin/sh

allennlp train experiments/semi_supervised/boe_trials/boe0.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe0
rm -rf experiments/semi_supervised/boe_trials/results/boe0/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe0/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe0/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe1.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe1
rm -rf experiments/semi_supervised/boe_trials/results/boe1/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe1/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe1/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe2.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe2
rm -rf experiments/semi_supervised/boe_trials/results/boe2/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe2/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe2/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe3.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe3
rm -rf experiments/semi_supervised/boe_trials/results/boe3/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe3/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe3/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe4.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe4
rm -rf experiments/semi_supervised/boe_trials/results/boe4/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe4/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe4/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe5.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe5
rm -rf experiments/semi_supervised/boe_trials/results/boe5/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe5/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe5/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe6.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe6
rm -rf experiments/semi_supervised/boe_trials/results/boe6/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe6/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe6/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe7.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe7
rm -rf experiments/semi_supervised/boe_trials/results/boe7/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe7/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe7/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe8.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe8
rm -rf experiments/semi_supervised/boe_trials/results/boe8/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe8/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe8/test.txt
allennlp train experiments/semi_supervised/boe_trials/boe9.json --include-package library -s experiments/semi_supervised/boe_trials/results/boe9
rm -rf experiments/semi_supervised/boe_trials/results/boe9/*.th
allennlp evaluate experiments/semi_supervised/boe_trials/results/boe9/model.tar.gz /data/dangt7/IMDB/test.filtered.jsonl --include-package library &>> experiments/semi_supervised/boe_trials/results/boe9/test.txt
