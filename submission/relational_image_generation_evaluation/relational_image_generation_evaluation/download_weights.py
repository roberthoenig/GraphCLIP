import gdown
import os



def download_weights(weights_name):
    destination = os.path.join(os.path.dirname(__file__), 'data', weights_name)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if weights_name == 'ViT-Base_Text_Emb_Hockey_Fighter.ckpt':
        # you can find the file id by opening the file in the browser in google drive and copying the id from the url
        file_id = '138GdG9GOlteVXU8s-1GC9sG9qy5HSOdK'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
    elif weights_name == 'ViT-Large_Text_Emb_Light_Sun.ckpt':
        file_id = '1wvNWKyPIM2Bldup_seNYLtdBbKT7vwTB'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
    elif weights_name == 'GraphCLIP.ckpt':
        # raise Exception()
        file_id = '1vhz_RRmdVhqSJ-ZC1y65-Lksd_zG-XM6'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
    elif weights_name == 'ViT-Large_Text_Emb_Spring_River.ckpt':
        file_id = '1a_m20zT5GjilVX7e_6tp8BtVzL9E2_Hc'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)

        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
    elif weights_name == 'hpc.pt':
        file_id = '1_KL-3i4CbDgKAAjvZu0qiDYd0Q0i1o0Z'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
    elif weights_name == 'ViT-Large_Text_Emb_Vocal_Snow.ckpt':
        file_id = '1V2MSpU8crjEvFWjmC9hjvwhR_Pio636p'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
    elif weights_name == 'histogram.ckpt':
        file_id = '1B6TRzzppsnwYZ9TrmKz4cTFJ_jJyWDFA'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
    elif weights_name == 'filtered_object_label_embeddings':
        text_embeddings_file_id = '1qRcI4XxGwjshiT8IOPf1dUP90JUHCsbE'
        text_embeddings_url = f'https://drive.google.com/uc?id={text_embeddings_file_id}'
        text_embeddings_destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_object_label_embeddings.pt')
        gdown.download(text_embeddings_url, text_embeddings_destination, quiet=False)
        

def build_filtered_graph(g, object_labels, relationship_labels, attribute_labels):
    G = nx.DiGraph()
    G.image_id=g.image_id
    G.image_w = g.image_w
    G.image_h = g.image_h
    G.labels = {}
    for n in g.nodes:
        if g.labels[n] in object_labels:
            filtered_attributes = [a for a in g.nodes[n]['attributes'] if a in attribute_labels]
            G.add_node(n, w=g.nodes[n]['w'], h=g.nodes[n]['h'], x=g.nodes[n]['x'], y=g.nodes[n]['y'], attributes=filtered_attributes, name=g.labels[n])
            G.labels[n] = g.labels[n]
    for e in g.edges:
        if g.edges[e]['predicate'] in relationship_labels and e[0] in G.nodes and e[1] in G.nodes:
            G.add_edge(e[0], e[1], synsets=g.edges[e]['synsets'].copy() ,relationship_id=g.edges[e]['relationship_id'], predicate=g.edges[e]['predicate'])
    return G

def filter_graphs(graphs):
    from .data import FILTERED_ATTRIBUTES, FILTERED_OBJECTS, FILTERED_RELATIONSHIPS
    import networkx as nx
    # filter the graphs relationships and objects and attributes to only keep the 100/200 most frequent ones. Remove graphs which have no objects or relationships left after filtering
    filtered_graphs = []
    for g in graphs:
        g_filtered = build_filtered_graph(g, FILTERED_OBJECTS, FILTERED_RELATIONSHIPS, FILTERED_ATTRIBUTES)
        if len(g_filtered.nodes) > 0 and len(g_filtered.edges) > 0:
            filtered_graphs.append(g_filtered)
    return filtered_graphs



def download_filtered_graphs():
    destination = os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs.pt')
    if not os.path.exists(destination):
        print('Downloading filtered graphs.')
        filed_id = '1xdQAjpATSAU-TR89CrqWp6aWXRctBDyz'
        url = f'https://drive.google.com/uc?id={filed_id}'
        gdown.download(url, destination, quiet=False)
    
    destination_test = os.path.join(os.path.dirname(__file__), 'data', 'filtered_graphs_test_small.pt')
    if not os.path.exists(destination_test):
        test_file_id = '1J2LQhXq8nvF4Bb1BW-I0zMmxdlANej7u'
        url = f'https://drive.google.com/uc?id={test_file_id}'
        gdown.download(url, destination_test, quiet=False)
    
    # import torch
    # graphs = torch.load(destination)
    # filtered_graphs = filter_graphs(graphs)
    # torch.save(filtered_graphs, destination)
    # graphs = torch.load(destination_test)
    # filtered_graphs = filter_graphs(graphs)
    # torch.save(filtered_graphs, destination_test)


def download_mscoco_graphs():
    destination = os.path.join(os.path.dirname(__file__), 'data', 'mscoco_graphs.pt')
    if not os.path.exists(destination):
        print('Downloading filtered graphs.')
        filed_id = '1Mu6dFN-5RsvA1pqIimgaHBJT1xeugEpa'
        url = f'https://drive.google.com/uc?id={filed_id}'
        gdown.download(url, destination, quiet=False)
    
    destination_test = os.path.join(os.path.dirname(__file__), 'data', 'mscoco_graphs_test_small.pt')
    if not os.path.exists(destination_test):
        test_file_id = '1MFLheYOGBC7Y-zHEv_o37ByyjdlBGy_s'
        url = f'https://drive.google.com/uc?id={test_file_id}'
        gdown.download(url, destination_test, quiet=False)


def download_cc500_graphs():
    destination = os.path.join(os.path.dirname(__file__), 'data', 'cc500_graphs.pt')
    if not os.path.exists(destination):
        print('Downloading filtered graphs.')
        filed_id = '15Stxq1B3qAgMdUt3qivAUozmgCzoq_j1'
        url = f'https://drive.google.com/uc?id={filed_id}'
        gdown.download(url, destination, quiet=False)
    
    destination_test = os.path.join(os.path.dirname(__file__), 'data', 'cc500_graphs_test_small.pt')
    if not os.path.exists(destination_test):
        test_file_id = '1Xruya-fFScPZPB5h5MK9ci6fdscukSlQ'
        url = f'https://drive.google.com/uc?id={test_file_id}'
        gdown.download(url, destination_test, quiet=False)

def download_adv_datasets():
    destination_relv1 = os.path.join(os.path.dirname(__file__), 'data', 'ra_selections_curated_adversarial.pt')
    if not os.path.exists(destination_relv1):
        file_id = '19d0wd5UjpvcUfroA_02XeUMkLnSD7hSx'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination_relv1, quiet=False)
    destination_relv2 = os.path.join(os.path.dirname(__file__), 'data', 'ra_selections_curated_adversarial2.pt')
    if not os.path.exists(destination_relv2):
        file_id = '1TsabJuArEOYNxMjBYNsezO3LkuPEo7Yd'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination_relv2, quiet=False)
    destination_attrv1 = os.path.join(os.path.dirname(__file__), 'data', 'realistic_adversarial_attributes_gt_accepted_pruned.json')
    if not os.path.exists(destination_attrv1):
        file_id = '1YwRAOoWPj0Bs3XUyaMW_F0Bc3L9wj1nP'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination_attrv1, quiet=False)