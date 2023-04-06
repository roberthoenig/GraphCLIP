import os
from pathlib import Path
import openai
# import tiktoken
from dotenv import load_dotenv
from tqdm import tqdm
import json
import torch
import shutil
import open_clip
import os.path as osp
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict 
import time

PROMPT = """
Please parse the following labeled scene graph into an equivalent human-readable sentence. Each line of the graph description lists a node's name, its attributes and its outgoing labeled edges to other nodes. Please only include information that is explicitly stated in the graph. Your description should contain all information in the graph and is not limited in length. The ordering of the edges and nodes is arbitrary, so they are all equally important.
Your description must omit the node identifier indices, that is, it must write "foo" instead of "foo (4)".
{}
"""

PROMPT2 = """
Please shorten the previous scene description such that it contains at most sixty words and retains as much information about the scene as possible. Write as terse as possible. Write in a telegraphic style, but don't abbreviate words and make sure to include information about the relation between objects in the scene with words like "next to", "behind", etc.
Also avoid mentioning the graph directly, that is, omit phrases like "In the scene", "this node has no attributes", etc.
"""

SCENE_GRAPHS_PATH = 'datasets/visual_genome/raw/scene_graphs.json'
IMAGE_DATA_PATH = 'datasets/visual_genome/raw/image_data.json'
MSCOCO_ANN_PATH = 'datasets/mscoco/annotations_trainval2017/annotations/captions_val2017.json'
OUT_JSON_PATH = 'scripts/chatgpt/captions.json'
OUT_IMG_DIR = 'scripts/chatgpt/images/'
VG_100K_DIR = 'datasets/visual_genome/raw/VG_100K'
VG_100K_2_DIR = 'datasets/visual_genome/raw/VG_100K_2'
N_CAPTION_SAMPLES = 10
N_CAPTIONS = 200
ID_PATH = 'datasets/mscoco/overlap.json'

def print_messages(messages):
    for m in messages:
        print('role', m['role'], 'content', m['content'])

def build_graph(g_dict):
    G = nx.DiGraph(image_id=g_dict['image_id'])
    G.labels = {}
    for obj in g_dict['objects']:
        G.add_node(obj['object_id'], w=obj['w'], h=obj['h'], x=obj['x'], y=obj['y'], attributes=obj.get('attributes',[]), name=obj['names'][0])
        G.labels[obj['object_id']] = obj['names'][0]
    for rel in g_dict['relationships']:
        G.add_edge(rel['subject_id'], rel['object_id'], synsets=rel['synsets'] ,relationship_id=rel['relationship_id'], predicate=rel['predicate'])
    return G
    
def plot_graph(g, idx):
    if g.number_of_nodes() == 0:
        plt.savefig(f"scripts/chatgpt/graphs/{idx}.png")
        plt.close()
    else:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        max_y = max([y for x,y in pos.values()])
        n_nodes_top = len([n for n in g.nodes if pos[n][1] == max_y])
        longest_label = max([len(g.labels[n]) for n in g.nodes])
        plt.figure(figsize=(max(n_nodes_top*longest_label/10,15),5))
        nx.draw(g,pos=pos,labels=g.labels, with_labels=True, node_size=10, node_color="lightgray", font_size=8)
        nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=nx.get_edge_attributes(g,'predicate'),font_size=8)
        plt.savefig(f"scripts/chatgpt/graphs/{idx}.png")
        plt.close()

def query_chatgpt(messages):
    # print_messages(messages)
    # exit(0)
    waiting_time = 1
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                n=1,
            )
            res = {
                "n_tokens": response["usage"]["total_tokens"],
                "caption": response.choices[0].message.content,
            }
            return res
        except Exception as e:
            print(f"Exception {e}.")
            print(f"Retrying in {waiting_time} seconds.")
            time.sleep(waiting_time)
            waiting_time *= 2

def graph_to_chatgpt_2(d):
    for idx, o in enumerate(d['objects']):
        o['name'] =  o['names'][0] + ' (' + str(idx) + ')'
    objs = {o['object_id']: o for o in d['objects']}
    edges = {s_id: [(r['predicate'], r['object_id']) for r in d['relationships'] if r['subject_id']==s_id] for s_id in objs.keys()}
    repr = []
    for o in d['objects']:
        name_repr = o['name']
        attr_repr = ','.join(o.get('attributes', []))
        # if edges[o['object_id']] == []:
            # o_repr = f"{name_repr} has attributes [{attr_repr}] and no outgoing edges"
        # else:
        edge_repr = ','.join([rel + " " + objs[id]['name'] for (rel,id) in edges[o['object_id']]])
        o_repr = f"{name_repr} [{attr_repr}] [{edge_repr}]"
        repr.append(o_repr)
    repr = '"""\n' + "\n".join(repr) + '\n"""'
    return repr

def graph_to_chatgpt(d):
    id_to_idx = {}
    # TODO: deal with multiple object names?
    n_obj_nodes = len(d['objects'])
    obj_names = [obj['names'][0] for obj in d['objects']]
    attrs = []
    for idx, o in enumerate(d['objects']):
        attr_list = o.get('attributes', [])
        attrs.append(attr_list)
    for idx, obj in enumerate(d['objects']):
        id_to_idx[obj['object_id']] = idx+1
    # edge_index: [2, num_edges]
    edge_index = torch.zeros((2, len(d['relationships'])), dtype=torch.long)
    edge_labels = []
    for ctr, rel in enumerate(d['relationships']):
        edge_index[:, ctr] = torch.tensor([id_to_idx[rel['subject_id']], id_to_idx[rel['object_id']]])
        edge_labels.append(rel['predicate'])
    adj_list = edge_index.t().tolist()
    adj_list_w_edge_labels = list(zip(adj_list, edge_labels))
    nodes = list(zip(range(1, n_obj_nodes+1), obj_names, attrs))
    res = {
        "node indices with node label and node attributes": nodes,
        "adjacency list with edge labels": adj_list_w_edge_labels,
    }
    return res

def graph_to_caption(d):
    n_tokens = 0
    chatgpt_graph = str(graph_to_chatgpt_2(d))
    # First message
    messages = [
        {"role": "user", "content": PROMPT.format(chatgpt_graph)}
    ]
    res = query_chatgpt(messages)
    n_tokens += res["n_tokens"]
    long_caption = res["caption"]
    print(f"({d['image_id']}) Long caption: {long_caption}")
    messages.append({"role": "assistant", "content": long_caption})
    # Second message
    short_caption = res["caption"]
    messages.append({"role": "user", "content": PROMPT2})
    res = query_chatgpt(messages)
    n_tokens += res["n_tokens"]
    short_caption = res["caption"]
    res = {"n_tokens": n_tokens, "long_caption": long_caption, "short_caption": short_caption}
    return res

def tok_to_usd(tok):
    return (tok / 1000) * 0.002

clip_tokenizer = open_clip.get_tokenizer(model_name="ViT-g-14")
def get_text_token_length(text):
    tokens = clip_tokenizer(text, context_length=1000)
    n_tokens = torch.sum(tokens!=0).item()
    return n_tokens

def get_text_word_length(text):
    return len(text.split())

def main():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open(SCENE_GRAPHS_PATH, 'r') as f:
        scene_graphs_dict = json.load(f)
    with open(IMAGE_DATA_PATH, 'r') as f:
        image_data_dict = json.load(f)
    with open(MSCOCO_ANN_PATH, 'r') as f:
        mscoco_ann_list = json.load(f)['annotations']
    mscoco_ann_dict = defaultdict(list)
    for d in mscoco_ann_list:
        mscoco_ann_dict[d['image_id']].append(d['caption'])
    mscoco_ann_dict = dict(mscoco_ann_dict)
    for d, i in zip(scene_graphs_dict, image_data_dict):
        d['coco_id'] = i['coco_id']
    
    image_id_to_path = dict()
    for dir in [Path(VG_100K_DIR), Path(VG_100K_2_DIR)]:
        pathlist = dir.glob('*.jpg')
        for path in pathlist:
            img_id = int(path.stem)
            image_id_to_path[img_id] = str(path)

    if osp.exists(OUT_JSON_PATH):
        with open(OUT_JSON_PATH, 'r') as f:
            out_list = json.load(f)
    else:
        out_list = []
    existing_coco_ids = [d['coco_id'] for d in out_list]
    with open(ID_PATH, 'r') as f:
        coco_overlap_ids = json.load(f)
    scene_graphs_filtered = [d for d in scene_graphs_dict if d['coco_id'] in coco_overlap_ids]
    scene_graphs_filtered = scene_graphs_filtered[:N_CAPTIONS]
    scene_graphs_filtered = [d for d in scene_graphs_filtered if d['coco_id'] not in existing_coco_ids]
    total_tokens = 0
    for d in tqdm(scene_graphs_filtered):
        image_id = d['image_id']
        coco_id = d['coco_id']
        if len(d['objects']) == 0:
            outs = {
                'n_tokens': 0
            }
            out_list.append({
                'image_id': image_id,
                'coco_id': coco_id,
                'captions': [""] * 10,
                'info': "Empty Graph!",
                'mscoco_captions': mscoco_ann_dict[coco_id],
            })
        else:
            outs = {
                'long_captions': [],
                'short_captions': [],
                'n_tokens': 0
            }
            for _ in range(N_CAPTION_SAMPLES):
                out = graph_to_caption(d)
                outs['long_captions'].append(out['long_caption'])
                outs['short_captions'].append(out['short_caption'])
                outs['n_tokens'] += out['n_tokens']
            n_tokens_long = [get_text_token_length(c) for c in outs['long_captions']]
            n_tokens_short = [get_text_token_length(c) for c in outs['short_captions']]
            captions =  [{
                    'long': c_long,
                    'short': c_short,
                    'n_tokens_long': n_tokens_long,
                    'n_tokens_short': n_tokens_short,
                } for (c_long, c_short, n_tokens_long, n_tokens_short) in zip(outs['long_captions'], outs['short_captions'], n_tokens_long, n_tokens_short)]
            captions.sort(key = lambda d: d["n_tokens_short"])
            out_list.append({
                'image_id': image_id,
                'coco_id': coco_id,
                'captions': captions,
                'mscoco_captions': mscoco_ann_dict[coco_id],
            })
            total_tokens += outs['n_tokens']
        # Captions
        with open(OUT_JSON_PATH, 'w') as f:
            json.dump(out_list, f)
        # Image
        shutil.copy(image_id_to_path[image_id], OUT_IMG_DIR)
        # Graph rendering
        plot_graph(build_graph(d), image_id)
        print(f"({image_id}) Tokens used for this example: {outs['n_tokens']} ({tok_to_usd(outs['n_tokens'])} USD)")
        print(f"({image_id}) Tokens used in total: {total_tokens} ({tok_to_usd(total_tokens)} USD)")
        print()

if __name__ == '__main__':
    main()