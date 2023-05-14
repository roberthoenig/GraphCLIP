import json
import os
from tkinter import Tk, Label, Button, Canvas
from PIL import Image, ImageTk
from tkinter import font

# Load newly generated graphs
with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt.json") as f:
    graphs = json.load(f)

# Initialize the list of accepted graphs
accepted_graphs = []

# Image directory paths
image_dir1 = "datasets/visual_genome/raw/VG_100K"
image_dir2 = "datasets/visual_genome/raw/VG_100K_2"

def show_graph(graph_index):
    if graph_index < len(graphs):
        graph = graphs[graph_index]
        image_id = graph["image_id"]

        # Update the counter label
        counter_text = f"Image ID: {image_id} | Processed: {graph_index + 1}/{len(graphs)} | Accepted: {len(accepted_graphs)}"
        counter_label.config(text=counter_text)

        # Check for image in both directories
        image_path1 = os.path.join(image_dir1, f"{image_id}.jpg")
        image_path2 = os.path.join(image_dir2, f"{image_id}.jpg")

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
        info_text = f"{e1['names'][0]} ({', '.join(e1['attributes'])}) - {e2['names'][0]} ({', '.join(e2['attributes'])})"
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
    with open("datasets/visual_genome/raw/realistic_adversarial_attributes_gt_accepted.json", "w") as f:
        # raise Exception()
        json.dump(accepted_graphs, f)

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
