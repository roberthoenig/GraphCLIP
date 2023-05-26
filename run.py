
import argparse
import logging
import pprint

from models.TextCLIP import TextCLIP
from models.GraphCLIP import GraphCLIP
from utils.experiment_utils import init_logger, prep_experiment_dir, load_experiment_config, init_seeds

import torch.multiprocessing

def main() -> None:
    torch.multiprocessing.set_sharing_strategy('file_system')
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
    elif config["model"] == "GraphCLIP":
        model = GraphCLIP(config)
    else:
        raise Exception(f"Unknown model {config['model']}")
    if config["type"] == "train":
        logging.info("Training model...")
        model.train()
    elif config["type"] == "eval":
        logging.info("Evaluating model...")
        model.eval()
    elif config["type"] == "eval_adversarial":
        logging.info("Evaluating model adversarially...")
        model.eval_adversarial()
    elif config["type"] == "eval_adversarial_attr":
        assert config["model"] == "TextCLIP"
        logging.info("Evaluating TextCLIP model adversarially for attributes...")
        model.eval_adversarial_attr()
    else:
        raise Exception(f"Unkown experiment type {config['type']}")

if __name__ == "__main__":
    main()
