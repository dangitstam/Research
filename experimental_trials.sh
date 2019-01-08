#!bin/sh

#######################################################
### Duplicate experiments with different random seeds.
#######################################################

# Semi-supervised experiments
python scripts/generate_random_seeds.py --config-path experiments/semi_supervised/boe.json --save-dir experiments/semi_supervised/boe_trials --omit-topic-printing
python scripts/generate_random_seeds.py --config-path experiments/semi_supervised/cnn.json --save-dir experiments/semi_supervised/cnn_trials --omit-topic-printing
python scripts/generate_random_seeds.py --config-path experiments/semi_supervised/rnn.json --save-dir experiments/semi_supervised/rnn_trials --omit-topic-printing

# Supervised experiments
python scripts/generate_random_seeds.py --config-path experiments/supervised/boe.json --save-dir experiments/supervised/boe_trials --omit-topic-printing
python scripts/generate_random_seeds.py --config-path experiments/supervised/cnn.json --save-dir experiments/supervised/cnn_trials --omit-topic-printing
python scripts/generate_random_seeds.py --config-path experiments/supervised/rnn.json --save-dir experiments/supervised/rnn_trials --omit-topic-printing

#######################################################
### Produce test results for each experiment
#######################################################

bash experiments/semi_supervised/boe_trials/run_experiments.sh
bash experiments/semi_supervised/cnn_trials/run_experiments.sh
bash experiments/semi_supervised/rnn_trials/run_experiments.sh

bash experiments/supervised/boe_trials/run_experiments.sh
bash experiments/supervised/cnn_trials/run_experiments.sh
bash experiments/supervised/rnn_trials/run_experiments.sh