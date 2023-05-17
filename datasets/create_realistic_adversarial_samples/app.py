from flask import Flask, render_template, request, send_file
import base64
import io
from PIL import Image
# from your_script import process_image, get_next_image
from utils import get_next_image, process_image, remove_edge

app = Flask(__name__ , template_folder='templates')

@app.route('/')
def index():
    print("getting next image")
    image_path, options = get_next_image()
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return render_template("index.html", img_data=img_data, options=options)

@app.route('/next_image', methods=["POST"])
def next_image():
    selected_options = request.form.getlist('options')
    process_image(selected_options)
    image_path, options = get_next_image()
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return render_template("index.html", img_data=img_data, options=options)

@app.route('/skip_edge', methods=["POST"])
def skip_edge():
    remove_edge()
    image_path, options = get_next_image()
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return render_template("index.html", img_data=img_data, options=options)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
