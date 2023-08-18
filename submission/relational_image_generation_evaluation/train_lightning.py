import relational_image_generation_evaluation as rige
from relational_image_generation_evaluation.vision_transformer.open_clip.jt_ViT_RelClassifier_lightning import ViT_RelClassifier
from pytorch_lightning.loggers import WandbLogger
import os
from datetime import datetime
from relational_image_generation_evaluation.vision_transformer.open_clip.transform import image_transform
from relational_image_generation_evaluation.vision_transformer.jt_training import CleanedVisualGenomeDataModule
import pytorch_lightning as pl
import torch

############################################################################
image_dir = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/raw/VG/"
run_logs_dir = "/local/home/jthomm/GraphCLIP/experiments/"

num_epochs = 10
clip_model_type =  'ViT-L-14' # 'ViT-L-14' #'ViT-g-14' #'ViT-H-14' #'ViT-B/32'
clip_pretrained_dataset = 'datacomp_xl_s13b_b90k' # 'laion2b_s32b_b82k' # 'laion2b_s34b_b88k'  # 'laion2b_s32b_b79k' # 'laion400m_e32'
shallow = True
debug_mode = True # turn this on such that a tiny dataset is loaded such that you can test the code
input_mode = 'text_embeddings' # 'bounding_boxes'
description = f"""
    ViT_RelClassifier with 100 epochs, 200 hidden size, and 128 batch size\n 
    clip model {clip_model_type}, clip pretrained dataset {clip_pretrained_dataset}
    Shallow CLassification Heads: {shallow}
    No Weighted Loss for the Rel Head and the Obj Heads
    Attribute Loss is enabled and the dataset is implemented with attributes
    Debug Mode: {debug_mode} (if true, this is only a tiny dataset for debugging purposes)
    Batch size: 128
    The adversarial dataset is removed from training and validation
    The data input mode is: {input_mode}
    Attribute loss is ignored for 0 attributes.
    Attribute loss weighting is turned OFF. (attention: attribute loss multiplier should be set accordingly)
    Learning rates increased to 5e-4 and 5e-6
    Object Heads are on, with loss
    """
############################################################################

### setup CUDA device visibility
torch.set_float32_matmul_precision('medium') # lightning recommends either 'medium' or 'high'
# free_devices = get_all_free_gpus_ids()
free_devices = [6]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in free_devices])
print(f"Making CUDA devices visible: {free_devices}")

### setup the logging and checkpointing
date_folder = run_logs_dir + datetime.now().strftime("%Y-%m-%d")
os.makedirs(date_folder, exist_ok=True)

run_id = len(os.listdir(date_folder))
run_name = f"vision_transformer_{run_id}"
run_folder = f"{date_folder}/{run_name}"
os.makedirs(run_folder)

wandb_logger = WandbLogger(project='ResearchAssistant', entity='jthomm', save_dir=run_folder, notes=description)

# Create the model
model = ViT_RelClassifier(
    100, 
    200, 
    100,
    clip_model_type, 
    clip_pretrained_dataset,
    shallow=shallow,
    mode=input_mode,
    )
image_size = model.ViT.image_size
print(f"Image size: {image_size}")
preprocess_function = image_transform(
    image_size,
    is_train=True,
    mean=None,
    std=None,
    aug_cfg=None,
)

# Create the data module
data_module = CleanedVisualGenomeDataModule(
        preprocess_function,
        metadata_path=None, # loaded automatically
        image_dir=image_dir,
        testing_only=debug_mode,
        batch_size=32,
        mode=input_mode,
)

### Weigthed Loss
# model.register_occurence_probabilities(
#     None,
#     None,
#     data_module.data[0].dataset.attr_occurence_probabilities,
#     logger=wandb_logger,
#     )

### Model Checkpointing
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    dirpath=run_folder,
    filename='best_rel_model',
    save_top_k=1,
    mode='max',
)
checkpoint_callback_every_epoch = pl.callbacks.ModelCheckpoint(
    dirpath=run_folder,
    filename='model_epoch',
    every_n_epochs=1,
    save_top_k=-1,
)


# Create the Trainer
trainer = pl.Trainer(
    max_epochs=num_epochs,
    logger=wandb_logger,
    accelerator="gpu" if torch.cuda.is_available() else 'cpu',
    devices = min(4, len(free_devices)) if torch.cuda.is_available() else 1,
    callbacks=[checkpoint_callback, checkpoint_callback_every_epoch],
    log_every_n_steps=1,
    accumulate_grad_batches=1,
    # strategy='ddp',
)

# Train the model
trainer.fit(model, data_module)
