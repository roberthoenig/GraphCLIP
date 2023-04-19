from argparse import ArgumentError
from typing import Optional, Tuple
from .visual_genome_dataset import get_dataloader

# import datasets
from pytorch_lightning import LightningDataModule


class CleanedVisualGenomeDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        preprocess_function,
        metadata_path,
        image_dir,
        testing_only=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.preprocess_function = preprocess_function
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.debug_mode = testing_only

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """
        self.dataloader_train, self.dataloader_val = get_dataloader(self.preprocess_function,self.metadata_path,self.image_dir, testing_only=self.debug_mode)


    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return self.dataloader_val

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.test_dataset,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         collate_fn=self.collate,
    #         shuffle=False,
    #     )