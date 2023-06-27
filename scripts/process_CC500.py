import json

def number_lines(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for i, line in enumerate(lines, start=1):
            f.write(f"{i} {line}")

def main(): 
    input_file = 'datasets/CC-500/CC-500_corrected.txt'
    output_file_txt = 'datasets/CC-500/CC-500_processed.txt'  
    input_file_json = 'datasets/CC-500/CC-500-difficult.json'
    output_file_json = 'datasets/CC-500/CC-500_corrected.json'
    output_file_captions = 'datasets/CC-500/CC-500_captions.json'
    
    number_lines(input_file, output_file_txt)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    graphs = []
    for id, line in enumerate(lines[:446], start=1):
        line = line.rstrip()
        if line == 'A green cup and a blue cell phone':
            attr1 = 'green'
            obj1 = 'cup'
            attr2 = 'blue'
            obj2 = 'cell phone'
        elif line == 'A blue cup and a green cell phone':
            attr1 = 'blue'
            obj1 = 'cup'
            attr2 = 'green'
            obj2 = 'cell phone'
        elif line == 'A red apple and yellow bananas':
            attr1 = 'red'
            obj1 = 'apple'
            attr2 = 'yellow'
            obj2 = 'bananas'
        elif line == 'A yellow apple and red bananas':
            attr1 = 'yellow'
            obj1 = 'apple'
            attr2 = 'red'
            obj2 = 'bananas'
        else:
            _, attr1, obj1, _, _, attr2, obj2 = line.split(' ')
        g = {
            "image_id": id,
            "relationships": [
            {
                "relationship_id": 1,
                "subject_id": 2,
                "predicate": "and",
                "object_id": 1
            },
            {
                "relationship_id": 2,
                "subject_id": 1,
                "predicate": "and",
                "object_id": 2
            }
            ],
            "objects": [
            {
                "object_id": 1,
                "names": [obj1],
                "attributes": [attr1]
            },
            {
                "object_id": 2,
                "names": [obj2],
                "attributes": [attr2]
            }
            ]
        }
        graphs.append(g)
    
    with open(input_file_json, 'r') as f:
        difficult_graphs = json.load(f)
    graphs += difficult_graphs
    
    # Check that there are no inconsistencies
    for id, graph in enumerate(graphs, start=1):
        if id != graph['image_id']:
            print(f"Invalid id at line {id}: {graph['image_id']}")
        for obj_id, obj in enumerate(graph['objects'], start=1):
            if obj_id != obj['object_id']:
                print(f"Invalid object id at line {id}: {obj_id}")
        obj_ids = [obj['object_id'] for obj in graph['objects']]
        for rel_id, rel in enumerate(graph['relationships'], start=1):
            if rel_id != rel['relationship_id']:
                print(f"Invalid relationship id at line {id}: {rel_id}")
            if rel['subject_id'] not in obj_ids:
                print(f"Invalid subject id {rel['subject_id']} at line {id}.")
            if rel['object_id'] not in obj_ids:
                print(f"Invalid object id {rel['object_id']} at line {id}.")
    
    with open(output_file_json, 'w') as f:
        json.dump(graphs, f, indent=1)

    captions = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line, graph in zip(lines, graphs):
        captions.append(
            {
                "image_id": graph['image_id'],
                "captions": {"short": line.rstrip()}
            }
        )
    with open(output_file_captions, 'w') as f:
        json.dump(captions, f, indent=1)
    
if __name__ == '__main__':
    main()