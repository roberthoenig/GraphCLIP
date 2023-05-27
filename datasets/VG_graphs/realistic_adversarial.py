import random
from . import utils
import torch




adv_perturbations = {
    'on': ['under', 'beside', 'next to', 'in front of', 'behind', 'above', 'below', 'in', 'inside'],
    'has': [],
    'in': ['outside', 'on'],
    'of': [],
    'wearing': [],
    'with': ['beside'],
    'behind': ['in front of', 'next to', 'beside', 'on'],
    'holding': ['carrying', 'has', 'using'],
    'on a': ['in a'],
    'near': ['next to', 'beside', 'on'],
    'on top of': ['under', 'beside', 'above'],
    'next to': ['on top of', 'under', 'in a', 'in', 'in front of', 'behind', 'on'],
    'has a': ['over'],
    'under': ['on', 'above', 'over', 'next to', 'beside', 'in front of', 'behind', 'on top of'],
    'of a': [],
    'by': ['wearing', 'covered in', 'worn by', 'wearing a'],
    'above': ['below', 'under', 'over', 'next to', 'beside', 'in front of', 'behind'],
    'wears': [],
    'in front of': ['behind', 'next to', 'beside', 'on'],
    'sitting on': ['standing on', 'laying on', 'lying on'],
    'on side of': ['next to', 'beside', 'along', 'on', 'in'],
    'attached to': [],
    'wearing a': [],
    'in a': ['on a', 'outside', 'next to', 'beside', 'in front of', 'behind', 'on'],
    'over': ['under', 'above', 'below'],
    'are on': ['are in', 'under', 'below', 'next to', 'beside', 'in front of', 'behind'],
    'at': ['beside', 'next to'],
    'for': [],
    'around': ['in', 'inside'],
    'beside': ['on top of', 'under', 'in a', 'in', 'in front of', 'behind', 'on'],
    'standing on': ['sitting on', 'laying on', 'lying on'],
    'riding': ['flying'],
    'standing in': ['sitting in', 'standing on'],
    'inside': ['outside'],
    'have': [],
    'hanging on': [],
    'walking on': ['standing on', 'sitting on', 'lying on'],
    'on front of': ['on back of', 'on top of', 'on side of'],
    'are in': ['are on', 'under', 'below', 'next to', 'beside', 'in front of', 'behind'],
    'hanging from': ['mounted on'],
    'carrying': ['flying', 'riding'],
    'holds': ['flying', 'riding'],
    'covering': [],
    'belonging to': [],
    'between': [],
    'along': [],
    'eating': [],
    'and': [],
    'sitting in': ['standing in', 'sitting on'],
    'watching': [],
    'below': ['above', 'over'],
    'painted on': ['sitting on'],
    'laying on': ['sitting on', 'standing on', 'lying on', 'walking on'],
    'against': ['supporting'],
    'playing': [],
    'from': [],
    'inside of': ['outside', 'next to', 'beside', 'in front of', 'behind'],
    'looking at': [],
    'with a': [],
    'parked on': [],
    'to': [],
    'has an': [],
    'made of': [],
    'covered in': [],
    'mounted on': ['hanging on'],
    'says': [],
    'growing on': [],
    'across': ['beside', 'next to'],
    'part of': [],
    'on back of': ['on front of', 'on side of'],
    'flying in': [],
    'outside': ['inside'],
    'lying on': ['sitting on', 'standing on'],
    'worn by': [],
    'walking in': [],
    'sitting at': ['lying on', 'standing on'],
    'printed on': [],
    'underneath': ['over', 'above', 'on'],
    'crossing': [],
    'beneath': ['under', 'below', 'above'],
    'full of': [],
    'using': [],
    'filled with': [],
    'hanging in': ['wearing'],
    'covered with': ['next to', 'beside'],
    'built into': [],
    'standing next to': ['in front of', 'behind', 'above', 'below'],
    'adorning': [],
    'a': [],
    'in middle of': [],
    'flying': [],
    'supporting': ['against'],
    'touching': [],
    'next': [],
    'swinging': [],
    'pulling': [],
    'growing in': [],
    'sitting on top of': ['lying on top of', 'walking on'],
    'standing': [],
    'lying on top of': ['sitting on top of', 'standing on', 'walking on'],
}


def get_relation_perturbation(predicate):
    predicate = predicate.lower().strip()
    if predicate in adv_perturbations and len(adv_perturbations[predicate]) > 0:
        return random.choice(adv_perturbations[predicate])
    else:
        return None

def get_relation_perturbations(predicate):
    predicate = predicate.lower().strip()
    if predicate in adv_perturbations and len(adv_perturbations[predicate]) > 0:
        return adv_perturbations[predicate]
    else:
        return None

def convert_adversarially_realistic(g, relationship_labels):
    '''returns a copy of the graph g with one edge replaced by a random edge from the relationship labels'''
    # convert to adversarial graph by looking for one edge that is in the relationship labels and replacing it with a random one from the relationship labels
    # we remove trailing spaces and convert to lower case
    g = utils.copy_graph(g)
    indexes = [i for i, e in enumerate(g.edges) if g.edges[e]['predicate'].strip().lower() in relationship_labels]
    random.shuffle(indexes)
    # now we try to convert the graph to adversarial and take it if not None
    for idx in indexes:
        adv_pert = get_relation_perturbation(g.edges[list(g.edges)[idx]]['predicate'])
        if adv_pert is not None:
            e = list(g.edges)[idx]
            relationship = (g.edges[e]['predicate'],adv_pert)
            g.edges[e]['predicate'] = adv_pert
            subject = g.nodes[e[0]]['name']
            object = g.nodes[e[1]]['name']
            return g, subject, object, relationship
    return g, None, None, None

def convert_all_adversarially_realistic(g, relationship_labels):
    '''returns a list of edges and all their adversarial perturbation options'''
    indexes = [i for i, e in enumerate(g.edges) if g.edges[e]['predicate'].strip().lower() in relationship_labels]
    adv_perturbations = []
    for idx in indexes:
        adv_perts = get_relation_perturbations(g.edges[list(g.edges)[idx]]['predicate'])
        if adv_perts is not None:
            e = list(g.edges)[idx]
            adv_perturbations.append((e, adv_perts))
    return adv_perturbations
    

def get_realistic_graphs_dataset(version, mode):
    if mode=='rel':
        return get_realistic_graphs_dataset_rel(version)
    elif mode=='attr':
        return get_realistic_graphs_dataset_attr(version)

def get_realistic_graphs_dataset_rel(version):
    metadata_path = utils.LOCAL_DATA_PATH +  "processed/"
    if version == 'v1':
        curated_adversarialt = torch.load(metadata_path + "ra_selections_curated_adversarial.pt") # a dict with image_id as key and a graph and the adversarial perturbations as value
    elif version == 'v2':
        curated_adversarialt = torch.load(metadata_path + "ra_selections_curated_adversarial2.pt")
    # the format of the dict is {image_id: [(original_graph,graph_edge,adv_predicate), ...]}
    # we return a list of tuples (graphs, adv_graph, adv_edge, adv_predicate) for each image for each adversarial perturbation
    dataset = []
    for original_graph, graph_edge, adv_predicate in curated_adversarialt:
        adv_graph = utils.copy_graph(original_graph)
        adv_graph.edges[graph_edge]['predicate'] = adv_predicate
        dataset.append({
            'original_graph': original_graph,
            'adv_graph': adv_graph,
            'changed_edge': graph_edge,
            'adv_predicate': adv_predicate
        })
    return dataset

def get_realistic_graphs_dataset_attr(version):
    import networkx as nx
    import json
    path = utils.LOCAL_DATA_PATH + 'raw/realistic_adversarial_attributes_gt_accepted_pruned.json'
    with open(path, 'r') as f:
        data = json.load(f)
        print(len(data))
    graphs = []
    graphs_adv = []
    for sample in data:
        graph = nx.DiGraph()
        graph.image_id = sample['image_id']
        obj1_name = sample['objects'][0]['names'][0]
        obj1_id = sample['objects'][0]['object_id']
        obj2_name = sample['objects'][1]['names'][0]
        obj2_id = sample['objects'][1]['object_id']
        obj1_attrs = sample['objects'][0]['attributes']
        obj2_attrs = sample['objects'][1]['attributes']
        # assert obj1_name in rige.FILTERED_OBJECTS, f"obj1_attrs: {obj1_attrs}"
        # assert obj2_name in rige.FILTERED_OBJECTS, f"obj2_attrs: {obj2_attrs}"
        # for attr in obj1_attrs:
        #     assert attr in rige.FILTERED_ATTRIBUTES, f"obj1_attrs: {obj1_attrs}"
        # for attr in obj2_attrs:
        #     assert attr in rige.FILTERED_ATTRIBUTES, f"obj2_attrs: {obj2_attrs}"
        # add nodes with attributes and labels
        graph.labels = {}
        graph.add_node(obj1_id, attributes=obj1_attrs, name=obj1_name)
        graph.labels[obj1_id] = obj1_name
        graph.add_node(obj2_id, attributes=obj2_attrs, name=obj2_name)
        graph.labels[obj2_id] = obj2_name
        graphs.append(graph)
        graph_adv = utils.copy_graph(graph)
        [n1,n2] = list(graph_adv.nodes)[0:2]
        graph_adv.nodes[n1]['attributes'], graph_adv.nodes[n2]['attributes'] = graph_adv.nodes[n2]['attributes'], graph_adv.nodes[n1]['attributes']
        graphs_adv.append(graph_adv)

    dataset = [{'original_graph': g, 'adv_graph': g_adv} for g, g_adv in zip(graphs, graphs_adv)]
    return dataset


