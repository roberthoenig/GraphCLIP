# Focus your Attention: Improving Text-to-Image Generation with Syntactical Restrictions

This project contains code for the paper `Focus your Attention: Improving Text-to-Image Generation with Syntactical Restrictions`. The following sections describe how to reproduce datasets, models
and experiments in the paper.

## Dataset: Difficult Adversarial Samples (DAA)

### How to use DAA

You can load and print the DAA dataset with the `rige` package as follows:
```python
import relational_image_generation_evaluation as rige
dataset_daa = rige.get_adversarial_attribute_dataset()
def get_prompt(sample):
    sid, oid = list(sample.edges)[0]
    rel = sample.edges[(sid, oid)]['predicate']
    os = sample.nodes[sid]['name']
    ats = sample.nodes[sid]['attributes']
    oo = sample.nodes[oid]['name']
    ato = sample.nodes[oid]['attributes']
    prompt = f"{','.join(ats)} {os} {rel} {','.join(ato)} {oo}".strip().lower()
    prompt += '.'
    return prompt
for sample in dataset_daa:
    print(f"Image ID: {sample['original_graph'].image_id}")
    print(f"Original Prompt: {get_prompt(sample['original_graph'])}")
    print(f"Adversarial Prompt: {get_prompt(sample['adv_graph'])}")
    print()
```
```
Image ID: 107907
Original Prompt: green grass and dead tree.
Adversarial Prompt: dead grass and green tree.

Image ID: 107907
Original Prompt: green grass and big tree.
Adversarial Prompt: big grass and green tree.

...
```

### How to reproduce DAA


We produce DAA by pre-filtering the set of all possible attribute-object-object-attribute
quadlets in `create_daa/prefilter.py`. We then use `create_daa/ui.py` to interactively select suitable
samples. To reproduce these steps, do the following:
0. Install the packages in `requirements.txt`.
1. Edit `prefilter.py` to set paths to appropriate Visual Genome dataset files.
2. Run `cd create_daa; python prefilter.py`. This produces `daa.json`.
3. Edit `ui.py` to set paths to appropriate Visual Genome dataset files, directories and to `daa.json`.
4. Run ```cd create_daa; python ui.py``` and use the UI to select suitable samples. This
   produces `daa_user_selected.json`.
5. Edit `ui.py` to enable the flag `FILTER_BY_is_adv_more_likely`. This will only present samples on which
   the histogram classifier fails.
6. Rerun ```cd create_daa; python ui.py``` and use the UI to select suitable samples. This
   updates `daa_user_selected.json`.
7. `daa_user_selected.json` is the final dataset.

## Experiment: Comparison between EPViT, CLIP and histogram

To reproduce the comparison of different text-image alignment evaluators,
run the notebook `compare_evaluators/compare_evaluators.ipynb`.