# Dataset creator for realistic adversarial relationship samples

A human selected small set of images and edges where exactly one edge is changed adversarially.
The graphs used here are filtered to only contain the top relationships (see the clean_visualgenome.ipynb notebook).

## How to use the dataset

 - add the file `ra_selections_curated_adversarial.pt` into `datasets/visual_genome/processed/`

 <!-- - make sure you created the files `filtered_* ` in the directory `datasets/visual_genome/processed`. If not, run the notebook clean_visualgenome.ipynb -->

 - check out `datasets/create_realistic_adversarial_samples/usage.ipynb` for how to use it.

## How to run to create more samples

 - add the files `ra_already_rated` and `ra_selections_curated_adversarial` into `datasets/visual_genome/processed/`

 - if you want to create from scratch you don't need to do that (i.e. delete those files)

 - change the paths in `datasets/create_realistic_adversarial_samples/utils.py` and `datasets/VG_graphs/utils.py` to your path 
 
  - make sure you created the files `filtered_* ` in the directory `datasets/visual_genome/processed`. If not, run the notebook clean_visualgenome.ipynb

 - open a commandline in the folder `datasts/create_realistic_adversarial_examples/`and run 
    ```python app.py ```

 - open the public URL shown in the commandline after loading successfully in your browser.