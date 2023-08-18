# Relational Image Generation Evaluation Package

This package aims at providing a simple interface of the EPViT, especially for use in Stable Diffusion. The model weights are loaded automatically, while the VisualGenome dataset (if you want to use it) needs to be downloaded separately.

## How to install

Navigate to the directory of this file in the commandline using your chosen python environment and type 

```
pip install -e .
```

## How to use

The package provides a dataset for evaluation and different models. 

The models available are:

* 'ViT-B/32' (finetuned ViT that takes in text embeddings of object names and classifies a fixed set of attributes and relationships)
* 'ViT-L/14' (larger finetuned ViT that takes in text embeddings of object names and classifies a fixed set of attributes and relationships). Optimally, use this model.
* 'ViT-L/14-Datacomp' It's the same model than ViT-L, but was pretrained with a newer, better dataset (see openclip). 

## example useage

**Please look at the usage.ipynb to get more extensive usage demonstration**

You can load a dataset and evaluate some images (here we just load visualgenome images) like this. The idea is that you generate images using stable diffusion to check whether they correspond to the graph.

```
import relational_image_generation_evaluation as rige

evaluator = rige.Evaluator('ViT-L/14')
# this is not the test set or so. It's just a flag to make loading faster
dataloader_two = rige.get_two_edge_dataloader()

from PIL import Image
images = []
graphs = []
for i in range(10):
    graph = next(iter(dataloader_two))[0]
    image_id = graph.image_id
    # adapt to your local directory
    IMAGE_DIR = '...youpath_to_visualgenome/VG/'
    image = Image.open(IMAGE_DIR + str(image_id) + '.jpg')
    images.append(image)
    graphs.append(graph)

scores = evaluator(images,graphs)
print(scores)
```

## Training
To train an EPViT model, run the train_lightning.py file in this directory (modify it to use the GPUs you want).