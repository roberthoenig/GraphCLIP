import json
import os
from tkinter import Tk, Label, Button, Canvas
from PIL import Image, ImageTk
from tkinter import font
from collections import defaultdict
import numpy as np
from copy import deepcopy

# Dataset paths
IMAGE_DIR1 = "datasets/visual_genome/raw/VG_100K"
IMAGE_DIR2 = "datasets/visual_genome/raw/VG_100K_2"
DATASET_INPUT_PATH = "datasets/visual_genome/raw/realistic_adversarial_attributes_gt.json"
DATASET_OUTPUT_PATH = "datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted_pruned.json"
FILTER_BY_is_adv_more_likely = True

# Load newly generated graphs
with open(DATASET_INPUT_PATH) as f:
    graphs = json.load(f)

# Initialize the list of accepted graphs
accepted_graphs = []

# Filter graphs based on histograms
def graph_to_strs(v):
    obj1 = v['objects'][0]
    obj1_txt = ','.join(obj1['attributes']) + '_' + obj1['names'][0]
    obj2 = v['objects'][1]
    obj2_txt = ','.join(obj2['attributes']) + '_' + obj2['names'][0]
    return [obj1_txt, obj2_txt]
strs_hist = defaultdict(int)
for g in graphs:
    for str in graph_to_strs(g):
        strs_hist[str] += 1

def is_adv_more_likely(g):
    g_adv = deepcopy(g)
    g_adv['objects'][0]['attributes'], g_adv['objects'][1]['attributes'] = g_adv['objects'][1]['attributes'], g_adv['objects'][0]['attributes']
    def get_p(v):
        v_strs = graph_to_strs(v)
        v_freqs = [strs_hist[s]+1 for s in v_strs]
        p = np.prod(v_freqs)
        return p
    p_gt = get_p(g)
    p_adv = get_p(g_adv)
    return p_adv >= p_gt  

with open(DATASET_OUTPUT_PATH, "r") as f:
    existing_graphs = json.load(f)
    existing_image_ids = [g['image_id'] for g in existing_graphs]

def not_existing_sample(graph):
    return graph['image_id'] not in existing_image_ids  

if FILTER_BY_is_adv_more_likely:
    graphs = list(filter(is_adv_more_likely, graphs))
graphs = list(filter(not_existing_sample, graphs))

def show_graph(graph_index):
    if graph_index < len(graphs):
        graph = graphs[graph_index]
        image_id = graph["image_id"]

        # Update the counter label
        counter_text = f"Image ID: {image_id} | Processed: {graph_index + 1}/{len(graphs)} | Accepted: {len(accepted_graphs)}"
        counter_label.config(text=counter_text)

        # Check for image in both directories
        image_path1 = os.path.join(IMAGE_DIR1, f"{image_id}.jpg")
        image_path2 = os.path.join(IMAGE_DIR2, f"{image_id}.jpg")

        image_path = image_path1 if os.path.exists(image_path1) else image_path2

        # Load image
        image = Image.open(image_path)
        image.thumbnail((1000, 1000))  # Resize image to twice the size

        # Update image on the canvas
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_image(0, 0, image=photo, anchor="nw")

        # Update object information
        e1, e2 = graph["objects"]
        info_text = f"{e1['names']} ({', '.join(e1['attributes'])}) - {e2['names']} ({', '.join(e2['attributes'])})"
        info_label.config(text=info_text)

        # Update buttons to show the next graph
        accept_button.config(command=lambda: accept_graph(graph_index))
        reject_button.config(command=lambda: reject_graph(graph_index))

def accept_graph(graph_index):
    global accepted_graphs
    accepted_graphs.append(graphs[graph_index])
    show_graph(graph_index + 1)

def reject_graph(graph_index):
    show_graph(graph_index + 1)

def save_accepted_graphs():
    with open(DATASET_OUTPUT_PATH, "w") as f:
        json.dump(existing_graphs + accepted_graphs, f)

# Create the main window
root = Tk()
root.title("Graph Review")

# Create the canvas to display the image
canvas = Canvas(root, width=1000, height=1000)
canvas.pack()

font = font.Font(size=24)
# Create the info label
info_label = Label(root, wraplength=500, font=font)
info_label.pack()

# Create the info label for the current image ID, processed samples, and accepted samples
counter_label = Label(root, font=font)
counter_label.pack()    

# Create the accept and reject buttons
accept_button = Button(root, text="Accept", command=lambda: accept_graph(0), font=font)
accept_button.pack(side="left")
reject_button = Button(root, text="Reject", command=lambda: reject_graph(0), font=font)
reject_button.pack(side="left")

# Create the save button
save_button = Button(root, text="Save", command=save_accepted_graphs,font=font)
save_button.pack(side="right")

# Display the first graph
show_graph(0)

# Start the main loop
root.mainloop()
