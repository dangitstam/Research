#!/bin/sh

DATA_PATH=$1
SAVE_PATH=$2

echo "Generating data mixtures:"
python scripts/generate_imdb_corpus.py --data-path $DATA_PATH --save-dir $SAVE_PATH

cd preprocessing

echo "Preprocessing all training mixtures:"
for i in 5 10 20
do
    TRAIN_PATH="../$SAVE_PATH/train_$i""k_labelled.jsonl"
    TRAIN_SAVE_PATH="../$SAVE_PATH/train_$i""k_labelled.filtered.jsonl"
    python filter_stopwords.py --data-path $TRAIN_PATH --save-path $TRAIN_SAVE_PATH
    mv "train_$i""k_labelled.bgfreq.json" ../data
done

# One more for the labelled case.
TRAIN_PATH="../$SAVE_PATH/train_labelled.jsonl"
TRAIN_SAVE_PATH="../$SAVE_PATH/train_labelled.filtered.jsonl"
python filter_stopwords.py --data-path $TRAIN_PATH --save-path $TRAIN_SAVE_PATH
mv "train_labelled.bgfreq.json" ../data

# Preprocess validation data.
VALID_PATH="../$SAVE_PATH/valid.jsonl"
VALID_SAVE_PATH="../$SAVE_PATH/valid.filtered.jsonl"
python filter_stopwords.py --data-path $VALID_PATH --save-path $VALID_SAVE_PATH
rm "valid.bgfreq.json"

# Preprocess test data.
VALID_PATH="../$SAVE_PATH/test.jsonl"
VALID_SAVE_PATH="../$SAVE_PATH/test.filtered.jsonl"
python filter_stopwords.py --data-path $VALID_PATH --save-path $VALID_SAVE_PATH
rm "test.bgfreq.json"