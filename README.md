# GraphCLIP

This repository explores a CLIP-like bi-encoder that matches images with scene graphs.

* `Prerequisites` describes how to set up your environment to run our experiments.

* `Usage` explains how to run our experiments.

## Prerequisites

### Software

0. Create a new Python 3.9.7 environment.

1. Install all packages in `requirements.txt` by running
   ```
   pip install -r requirements.txt
   ```

### Datasets

#### MSCOCO

1. Download http://images.cocodataset.org/zips/val2017.zip and extract it into
   `datasets/mscoco`.
   The images are in `datasets/mscoco/val2017/\d+.jpg`

2. Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   and extract it into `datasets/mscoco`.

## Usage

The directory `experiments` stores the specification of each experiment in a `.toml` config file.

To run experiment `experiments/<EXPERIMENT>.toml`, execute

```sh
python run.py <EXPERIMENT>
```

The previous command logs results to `experiments//<EXPERIMENT>/<RUN_NUMBER>`.
