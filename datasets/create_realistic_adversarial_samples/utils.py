# import ..VG_graphs as VG_graphs
import sys
sys.path.append("..")
import VG_graphs
from VG_graphs import get_filtered_graphs, get_filtered_relationships, get_filtered_objects, get_filtered_attributes, copy_graph
import os
from VG_graphs.realistic_adversarial import convert_all_adversarially_realistic
from PIL import Image
import random
import torch

filtered_graphs = get_filtered_graphs(False)
relationship_labels = get_filtered_relationships()
object_labels = get_filtered_objects()
attribute_labels = get_filtered_attributes()

random.seed(42)

shuffled_graphs = filtered_graphs.copy()
random.shuffle(shuffled_graphs)

####################################################################################
image_dir = "/local/home/stuff/visual-genome/VG/"
metadata_path = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/"
####################################################################################

try:
    selections = torch.load(metadata_path + "ra_selections_curated_adversarial.pt") # a dict with image_id as key and a graph and the adversarial perturbations as value
    # shuffled_graphs = [g for g in shuffled_graphs if g.image_id not in curated_adversarial.keys()]
    print(f"loaded {len(selections)} many selections")
except:
    print("creating new selections")
    selections = []

try:
    already_rated = torch.load(metadata_path + "ra_already_rated.pt") # a dict with image_id as key and a graph and the adversarial perturbations as value
    print(f"loaded {len(already_rated)} many already rated")
except:
    already_rated = set()

# The list of image paths, in the same order as the graphs
image_path_list = [image_dir + f'{g.image_id}.jpg' for g in shuffled_graphs]

current_graph_index = 0

def _options_list(index):
    g = shuffled_graphs[index]
    g = copy_graph(g) # to make sure each selected option is a different graph
    adv_perturbations = convert_all_adversarially_realistic(g, relationship_labels)
    res = []
    for adv_perturbation in adv_perturbations:
        graph_edge = adv_perturbation[0]
        subject = g.nodes[graph_edge[0]]['name']
        object = g.nodes[graph_edge[1]]['name']
        predicate = g.edges[graph_edge]['predicate']
        adv_predicates = adv_perturbation[1]
        for adv_predicate in adv_predicates:
            res.append((f'{subject} {predicate} {object} -> {subject} {adv_predicate} {object}',(graph_edge,adv_predicate)))
            # res.append((f'{subject} {adv_predicate} {object}',(g,graph_edge,adv_predicate)))
        random.shuffle(res)
    return res[:30]

# the options metadata when it comes back has the format: "((1150105, 1150094), 'on top of')"
def _parse_selection(s):
    edge = s[s[1:].find("(")+2:s.find(")")]
    edge = edge.split(",")
    edge = (int(edge[0]), int(edge[1]))
    adv_predicate = s[s.find("'")+1:s.rfind("'")]
    return edge, adv_predicate

def get_next_image():
    # Load the next image and options
    # ...

    global current_graph_index
    global image_path_list
    global shuffled_graphs
    while current_graph_index + 1 < len(image_path_list) and shuffled_graphs[current_graph_index].image_id in already_rated:
        current_graph_index += 1
            
    if current_graph_index + 1 == len(image_path_list):
        print("Finished!")
        return "", []
    image_path = image_path_list[current_graph_index]
    options = _options_list(current_graph_index)
    return image_path, options



def process_image(selected_options):
    # Process the selected options and update the data structures
    # ...
    global already_rated
    global current_graph_index
    already_rated.add(shuffled_graphs[current_graph_index].image_id)
    global selections
    new_selections = []
    for s in selected_options:
        edge, adv_predicate = _parse_selection(s)
        g = copy_graph(shuffled_graphs[current_graph_index])
        selections.append((g, edge, adv_predicate))
        new_selections.append((g, edge, adv_predicate))
    print(new_selections)
    torch.save(selections, metadata_path + "ra_selections_curated_adversarial.pt")
    torch.save(already_rated, metadata_path + "ra_already_rated.pt")
    print(f"saved {len(selections)} many selections")

    # next round
    current_graph_index += 1



