import json


with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted.json", "r") as f:
    gt = json.load(f)
    
adv = []
for g in gt:
    g['objects'][0]['attributes'], g['objects'][1]['attributes'] = g['objects'][1]['attributes'], g['objects'][0]['attributes']
    adv.append(g)
    
with open("datasets/visual_genome/raw/realistic_adversarial_attributes_adv_accepted.json", "w") as f:
    json.dump(adv, f)