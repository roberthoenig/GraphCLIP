from setuptools import setup, find_packages

setup(
    name="relational_image_generation_evaluation",
    version="1.0",
    description="This package aims at bringing Roberts and my models and datasets to Stable Diffusion, making our evaluation methods easily accessible.",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'ftfy',
        'numpy',
        'scikit-learn',
        'networkx>=3.0',
        'gdown',
        'huggingface_hub',
        'Pillow',
        'tqdm',
        'pytorch_lightning',
        'torchmetrics',
        'transformers',
        'matplotlib',
        'wandb',
        'open_clip_torch',
        'torch_geometric',
        'requests',
        # Add all the packages your code depends on
        # They will be installed via pip when your package is installed
    ],
)