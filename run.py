
import argparse
import logging
import pprint

from models.TextCLIP import TextCLIP
from utils.experiment_utils import init_logger, prep_experiment_dir, load_experiment_config, init_seeds

def main() -> None:
    parser = argparse.ArgumentParser(description=
"""
Run an experiment.
""")
    parser.add_argument( "experiment", type=str,
        help=f"Name of the experiment that you want to run.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.experiment)
    experiment_dir = prep_experiment_dir(args.experiment)
    config["experiment_dir"] = experiment_dir
    
    init_seeds(config["seed"])
    
    init_logger(str(experiment_dir / 'output.log'))

    logging.info(pprint.pformat(config))

    if config["model"] == "TextCLIP":
        model = TextCLIP(config)
    else:
        raise Exception(f"Unknown model {config['model']}")
    if config["type"] == "train":
        logging.info("Training model...")
        model.train()
    elif config["type"] == "eval":
        logging.info("Evaluating model...")
        model.eval()
    else:
        raise Exception(f"Unkown experiment type {config['type']}")

if __name__ == "__main__":
    main()
