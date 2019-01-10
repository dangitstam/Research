import argparse
import json
import os
import shutil
import random


def main():
    """
    Given a config file, produce a directory containing copies of that
    config file with randomly generated random seeds.

    TODO: Include a bash script that runs each of these one at a time.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--config-path", type=str,
                        help="Path to the config file.")
    parser.add_argument("--save-dir", type=str,
                        default=project_root,
                        help="Directory to store the randomly seeded configs.")
    parser.add_argument("--omit-topic-printing", action='store_true',
                        help="For convenience, overrides a config file's `print_topics` flag.")
    parser.add_argument("--num-seeds", type=int,
                        default=10,
                        help="Number of randomly seeded configs to generate.")
    args = parser.parse_args()

    config_basename = os.path.basename(args.config_path)
    config_json = json.load(open(args.config_path, "r"))

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)

    os.mkdir(args.save_dir)

    results_dir = os.path.join(args.save_dir, "results")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    script = "#!/bin/sh\n\n"

    def save_randomly_seeded_config(config_json, iteration):
        nonlocal script

        experiment_name = config_basename.split(".")[0] + str(iteration) 
        new_config_basename = experiment_name + ".json"
        new_config_path = os.path.join(args.save_dir, new_config_basename)
        new_config_file = open(new_config_path, "w")
        json.dump(config_json, new_config_file, indent=2)
        new_config_file.close()

        # Write to the bash script.
        command = "allennlp train {} --include-package library -s {}\n".format(
            new_config_path,
            os.path.join(results_dir, experiment_name)
        )

        script += command

        # For saving space; particularly good models can always be
        # re-trained.
        command = "rm -rf {}/*.th\n".format(
            os.path.join(results_dir, experiment_name)
        )
        script += command

        # Evaluate the model on test.
        command = ("allennlp evaluate {}/model.tar.gz "
                   "/data/dangt7/IMDB/test.filtered.jsonl "
                   "--include-package library &>> {}\n").format(
            os.path.join(results_dir, experiment_name),
            os.path.join(results_dir, experiment_name, "test.txt")
        )

        script += command

    # Each new config file will be suffixed with a number from 0 to
    # (num_seeds - 1).
    for seed in range(args.num_seeds):
        random_seed = random.randint(0, 10000)
        numpy_seed = random.randint(0, 10000)
        pytorch_seed = random.randint(0, 10000)

        config_json_randomly_seeded = config_json
        config_json_randomly_seeded["random_seed"] = random_seed
        config_json_randomly_seeded["numpy_seed"] = numpy_seed
        config_json_randomly_seeded["pytorch_seed"] = pytorch_seed

        # Exclude printing of topics.
        if args.omit_topic_printing:
            config_json_randomly_seeded["model"]["print_topics"] = False

        save_randomly_seeded_config(config_json_randomly_seeded, seed)

    with open(os.path.join(args.save_dir, "run_experiments.sh"), "w") as f:
        f.write(script)
        f.close()


if __name__ == "__main__":
    main()
