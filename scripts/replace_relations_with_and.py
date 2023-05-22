import json

for file in ['realistic_adversarial_attributes_gt_accepted_pruned', 'realistic_adversarial_attributes_adv_accepted_pruned']:
    with open(f"datasets/visual_genome/raw/{file}.json", "r") as f:
        graphs = json.load(f)
    print("len(graphs)", len(graphs))
    for graph in graphs:
        e1, e2 = graph['objects']
        graph['relationships'] = [
            {
                "predicate": "and",
                "object_id": e1['object_id'],
                "subject_id": e2['object_id'],
            },
            {
                "predicate": "and",
                "object_id": e2['object_id'],
                "subject_id": e1['object_id'],
            },
        ]
    with open(f"datasets/visual_genome/raw/{file}_and.json", "w") as f:
        json.dump(graphs, f)
  