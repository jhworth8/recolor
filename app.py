from flask import Flask, render_template, request, send_from_directory, flash, redirect, jsonify
import cv2
from flask_cors import CORS
import numpy as np
import os
from PIL import Image, ExifTags
from werkzeug.utils import secure_filename
import tensorflow_hub as hub
import tensorflow as tf
import io
import time
import requests
from googleapiclient.discovery import build
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
STYLE_TRANSFER_MODEL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'

# Load the pre-trained model and the model configuration
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
pts_in_hull = np.load('pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

# Load Style Transfer model
style_transfer_model = hub.load(STYLE_TRANSFER_MODEL)

GOOGLE_API_KEY = 'AIzaSyCCQAnc1GFCj0ZErdBjC8Qpv4tSkzw6aT4'
GOOGLE_CSE_ID = '970835fe6194d4ed0'



CORS(app, resources={r"/search": {"origins": "*"}})
@app.route('/search')
def search_images():
    query = request.args.get('query', '')
    if query:
        # Call the Google Custom Search API and parse the results
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&searchType=image&key={GOOGLE_API_KEY}&num=1"
        response = requests.get(search_url)
        search_results = response.json()
        image_urls = [item['link'] for item in search_results.get('items', [])]
        return jsonify(image_urls)
    else:
        return jsonify([])
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    content_file = request.files.get('content')

    if not content_file:
        flash('No content file selected')
        return redirect(request.url)

    if not content_file.content_type.startswith('image'):
        flash('Content file is not an image')
        return redirect(request.url)

    content_filename = secure_filename(content_file.filename)
    content_filepath = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    content_file.save(content_filepath)

    stylized_img_path = None

    style_file = request.files.get('style')
    style_keyword = request.form.get('style_keyword')  # Retrieve the keyword from the form

    # If a style file is provided, perform a style transfer directly on the original image.
    if style_file and style_file.content_type.startswith('image'):
        style_filename = secure_filename(style_file.filename)
        style_filepath = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        style_file.save(style_filepath)

        stylized_img_path = style_transfer(content_filepath, style_filepath)

    # If a keyword is provided instead of an image, search, download the image, and then perform style transfer.
    elif style_keyword:
        style_filepath = search_and_download(style_keyword)

        if style_filepath:  # If the image download was successful
            stylized_img_path = style_transfer(content_filepath, style_filepath)

    # If a style file or keyword is provided and style transfer succeeded, show the result.
    if stylized_img_path:
        return render_template('index.html', input_image=content_filename,
                               style_image=os.path.basename(style_filepath if style_file else style_filepath),
                               output_image=os.path.basename(stylized_img_path))

    # If neither a style file nor a keyword is provided, convert to black and white, then colorize.
    else:
        # Convert the original image to black and white first
        bw_image_path = convert_to_bw(content_filepath)
        # Now colorize the black and white image
        colorized_img_path = colorize_image(bw_image_path)
        return render_template('index.html', input_image=content_filename, output_image=os.path.basename(colorized_img_path))
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def colorized_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def convert_to_bw(filepath):
    img = cv2.imread(filepath)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filepath, bw_img)  
    # Overwrite the original image with its black and white version
    return filepath

def colorize_image(filepath):
    img = cv2.imread(filepath, 1)
    img_rgb = img.astype(np.float32) / 255.
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]
    img_l_rs = cv2.resize(img_l, (224, 224))
    img_l_rs -= 50
    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_dec_us = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
    img_bgr_out = cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR)
    img_bgr_out = np.clip(img_bgr_out * 255, 0, 255).astype(np.uint8)
    # Generate a unique filename
    timestamp = int(time.time())
    original_filename = os.path.basename(filepath)
    unique_filename = f"colorized_{original_filename.split('.')[0]}_{timestamp}.jpg"

    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)
    cv2.imwrite(output_filepath, img_bgr_out)
    return output_filepath

def style_transfer(content_path, style_path, style_max_dim=200):
    # Ensure the model is correctly named to avoid recursive function calls
    global style_transfer_model
    content_img = load_img(content_path)
    style_img = load_img(style_path, max_dim=style_max_dim)
    
    # Perform style transfer using the TensorFlow Hub model
    stylized_img = style_transfer_model(tf.constant(content_img), tf.constant(style_img))[0]
    
    # Convert the result tensor to a numpy array and remove the extra dimension
    stylized_img_array = np.squeeze(stylized_img.numpy())
    
    # Generate a unique filename
    timestamp = int(time.time())
    original_filename = os.path.basename(content_path)
    unique_filename = f"stylized_{original_filename.split('.')[0]}_{timestamp}.jpg"

    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], unique_filename)

    tf.keras.preprocessing.image.save_img(output_filepath, stylized_img_array)
    return output_filepath

def search_and_download(search_term):
    # Function to perform a search and download an image
    for start_index in range(1, 11, 10): 
        params = {
            'q': search_term,
            'cx': GOOGLE_CSE_ID,
            'key': GOOGLE_API_KEY,
            'searchType': 'image',
            'start': start_index,
            'num': 2,
            'imgSize': 'large',
        }

        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        response_json = response.json()

        if 'items' in response_json:
            for item in response_json['items']:
                image_url = item['link']
                style_image_response = requests.get(image_url)

                if style_image_response.status_code == 200:
                    # Save the image to the uploads directory
                    style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style_' + search_term + '_' + str(start_index) + '.jpg')
                    with open(style_image_path, 'wb') as f:
                        f.write(style_image_response.content)
                    return style_image_path

    return None

def load_img(path_to_img, max_dim=1024):
    img = tf.io.read_file(path_to_img)

    if path_to_img.lower().endswith('.webp'):  
        # Handle WEBP files
        with Image.open(io.BytesIO(img.numpy())) as image:
            with io.BytesIO() as jpeg_io:
                image.convert('RGB').save(jpeg_io, 'JPEG')
                jpeg_io.seek(0)
                img = jpeg_io.read()

    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]
    return img


if __name__ == '__main__':
    app.run(debug=True)