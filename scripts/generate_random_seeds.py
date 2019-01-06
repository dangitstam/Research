import argparse
import json
import os
import random


def main():
    """
    Given a config file, produce a directory containing copies of that
    config file with randomly generated random seeds.
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
    parser.add_argument("--num-seeds", type=int,
                        default=10,
                        help="Number of randomly seeded configs to generate.")
    args = parser.parse_args()

    config_basename = os.path.basename(args.config_path)
    config_json = json.load(open(args.config_path, "r"))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    def save_randomly_seeded_config(config_json, iteration):
        new_config_basename = (config_basename.split(".")[0] +
                               str(iteration) + ".json")
        new_config_path = os.path.join(args.save_dir, new_config_basename)
        new_config_file = open(new_config_path, "w")
        json.dump(config_json, new_config_file, indent=2)
        new_config_file.close()       

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

        save_randomly_seeded_config(config_json_randomly_seeded, seed)


if __name__ == "__main__":
    main()
