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
import networkx as nx

####################################################################################
image_dir = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/raw/VG/"
metadata_path = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/"
MODE = "unique-triplets" #"multi-predicate"
####################################################################################

filtered_graphs_ = get_filtered_graphs(False)
relationship_labels = get_filtered_relationships()
object_labels = get_filtered_objects()
attribute_labels = get_filtered_attributes()

# random.seed(423)

shuffled_graphs_ = filtered_graphs_.copy()
random.shuffle(shuffled_graphs_)

try:
    already_rated = torch.load(metadata_path + "ra_already_rated2.pt") # a dict with image_id as key and a graph and the adversarial perturbations as value
    print(f"loaded {len(already_rated)} many already rated")
except:
    already_rated = set()


shuffled_graphs = []
# filter out objects that are too small
for g in shuffled_graphs_:
    graph_copy = copy_graph(g)
    width, height = g.image_w, g.image_h
    for node in g.nodes:
        relative_width = g.nodes[node]['w'] / width
        relative_height = g.nodes[node]['h'] / height
        # remove small objects and objects that are too close to the border, because they are probably background
        background_score = g.nodes[node]['x']-width*0.1 >=0 
        background_score += g.nodes[node]['y']-height*0.1 >=0 
        background_score += g.nodes[node]['x']+g.nodes[node]['w']+width*0.1 <= width 
        background_score += g.nodes[node]['y']+g.nodes[node]['h']+height*0.1 <= height
        if relative_width*relative_height > 0.1 and background_score < 3:
            pass
        else:
            graph_copy.remove_node(node)
    if len(graph_copy.edges) > 0:
        shuffled_graphs.append(graph_copy)

print(f"filtered out {len(shuffled_graphs_)-len(shuffled_graphs)} graphs because their objects were too small or too close to the border, or already rated. {len(shuffled_graphs)} many graphs remain")

histogram = {}
for graph in shuffled_graphs:
    for edge in graph.edges:
        key = graph.nodes[edge[0]]['name'] + " " + graph.nodes[edge[1]]['name']
        if key not in histogram:
            histogram[key] = {graph.edges[edge]['predicate']: [(graph, edge)]}
        elif MODE == "multi-predicate":
            if graph.edges[edge]['predicate'] not in histogram[key]:
                histogram[key][graph.edges[edge]['predicate']] = [(graph, edge)]
            else:
                histogram[key][graph.edges[edge]['predicate']].append((graph, edge))
        elif MODE == "unique-triplets":
            histogram.pop(key) # remove the key, because there is more than one occurence
list_of_keys = list(histogram.keys())
if MODE == "multi-predicate":
    for key in list_of_keys:
        if len(histogram[key].keys()) <= 1:
            # drop the key, because there is only one predicate
            histogram.pop(key)
elif MODE == "unique-triplets":
    pass
else:
    raise Exception("unknown mode")

print(f"created histogram with {len(histogram)} many keys")
# choose a random key from the histogram
state = {
    "key": None,
    "already_chosen_predicate": None,
    "current_graph": None,
    "current_edge": None,
}

try:
    selections = torch.load(metadata_path + "ra_selections_curated_adversarial2.pt") # a dict with image_id as key and a graph and the adversarial perturbations as value
    # shuffled_graphs = [g for g in shuffled_graphs if g.image_id not in curated_adversarial.keys()]
    print(f"loaded {len(selections)} many selections")
except:
    print("creating new selections")
    selections = []

# The list of image paths, in the same order as the graphs
image_paths = [image_dir + f'{g.image_id}.jpg' for g in shuffled_graphs]

def _options_list(g):
    g = copy_graph(g) # to make sure each selected option is a different graph
    adv_perturbations = convert_all_adversarially_realistic(g, relationship_labels)
    res = []
    for adv_perturbation in adv_perturbations:
        graph_edge = adv_perturbation[0]
        if graph_edge != state["current_edge"]:
            continue
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

    global shuffled_graphs
    global state
    global histogram
    # while current_graph_index + 1 < len(image_path_list) and shuffled_graphs[current_graph_index].image_id in already_rated:
    #     current_graph_index += 1
    
            
    # if current_graph_index + 1 == len(image_path_list):
    #     print("Finished!")
    #     return "", []
    if state["key"] is None or state["key"] not in histogram:
        new_key = random.choice(list(histogram.keys()))
        next_predicate = random.choice(list(histogram[new_key].keys()))
        next_graph_and_edge = random.choice(histogram[new_key][next_predicate])
        state = {
            "key": new_key,
            "already_chosen_predicate": None,
            "already_chosen_adv_predicate": None,
            "already_chosen_graph_and_edge": None,
            "current_graph": next_graph_and_edge[0],
            "current_edge": next_graph_and_edge[1],
        }
    else:
        if state["already_chosen_predicate"] is None:
            next_predicate = random.choice(list(histogram[state["key"]].keys()))
        else:
            next_predicate = random.choice(list(set(histogram[state["key"]].keys())))
        next_graph_and_edge = random.choice(histogram[state["key"]][next_predicate])
        state = {
            "key": state["key"],
            "already_chosen_predicate": state["already_chosen_predicate"],
            "already_chosen_adv_predicate": state["already_chosen_adv_predicate"],
            "already_chosen_graph_and_edge": state["already_chosen_graph_and_edge"],
            "current_graph": next_graph_and_edge[0],
            "current_edge": next_graph_and_edge[1],
        }


    image_path = image_dir + f'{state["current_graph"].image_id}.jpg'
    options = _options_list(state["current_graph"])
    if len(options) == 0:
        current_predicate = state["current_graph"].edges[state["current_edge"]]["predicate"]
        new_predicate_list = [x for x in histogram[state["key"]][current_predicate] if x[1]!=state["current_edge"]]
        histogram[state["key"]][current_predicate] = new_predicate_list
        if len(histogram[state["key"]][current_predicate]) == 0:
            histogram[state["key"]].pop(current_predicate)
        if state["key"] is not None and len(histogram[state["key"]]) == 0:
            histogram.pop(state["key"])
        return get_next_image()
    return image_path, options



def process_image(selected_options):
    # Process the selected options and update the data structures
    # ...
    global state
    global histogram
    already_rated.add((state["current_graph"].image_id, state["current_edge"]))
    global selections
    if len(selected_options) > 1:
        print("ERROR: more than one option selected, only the first one will be saved")
    if len(selected_options) > 0:
        s = selected_options[0]
        edge, adv_predicate = _parse_selection(s)
        orig_predicate = state["current_graph"].edges[edge]["predicate"]
        g = copy_graph(state["current_graph"])
        if MODE == "multi-predicate":
            if state["already_chosen_graph_and_edge"] is not None:
                selections.append((state["already_chosen_graph_and_edge"][0], state["already_chosen_graph_and_edge"][1], state["already_chosen_adv_predicate"]))
                selections.append((g, edge, adv_predicate))
                print(f"saved {state['already_chosen_graph_and_edge'][0].image_id} with {state['already_chosen_graph_and_edge'][1]} and {state['already_chosen_adv_predicate']}")
                print(f"saved {g.image_id} with {edge} and {adv_predicate}")
                state["key"] = None
            else:
                state["already_chosen_graph_and_edge"] = (g, edge)
                state["already_chosen_predicate"] = orig_predicate
                state["already_chosen_adv_predicate"] = adv_predicate
                print(f"added candidate {g.image_id} with {edge} and {adv_predicate}")
                assert state["key"] in histogram
                histogram[state["key"]].pop(orig_predicate)
        elif MODE=="unique-triplets":
            selections.append((g, edge, adv_predicate))
            histogram.pop(state["key"]) # since we are in the unique node, we can remove the key
            print(f"saved {g.image_id} with {edge} and {adv_predicate}")
    else:
        # remove the current graph, edge from the histogram
        current_predicate = state["current_graph"].edges[state["current_edge"]]["predicate"]
        new_predicate_list = [x for x in histogram[state["key"]][current_predicate] if x[1]!=state["current_edge"]]
        assert len(new_predicate_list) == len(histogram[state["key"]][current_predicate])-1
        histogram[state["key"]][current_predicate] = new_predicate_list
        if len(histogram[state["key"]][current_predicate]) == 0:
            histogram[state["key"]].pop(current_predicate)
    if state["key"] is not None and state["key"] in histogram and len(histogram[state["key"]]) == 0:
        histogram.pop(state["key"])

    torch.save(selections, metadata_path + "ra_selections_curated_adversarial2.pt")
    torch.save(already_rated, metadata_path + "ra_already_rated2.pt")
    print(f"saved {len(selections)} many selections")


def remove_edge():
    global state
    global histogram
    histogram.pop(state["key"])
