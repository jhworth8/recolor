from flask import Flask, render_template, request, send_from_directory, flash, redirect
import cv2
import numpy as np
import os
from PIL import Image, ExifTags
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key'
# Load the pre-trained model and the model configuration
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')

# Load the cluster centers used for colorization
pts_in_hull = np.load('pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

# Assign the blobs
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['file']

    # Check if the file has a filename (indicating a file was selected)
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Ensure the file is an image
    if not file.content_type.startswith('image'):
        flash('File is not an image')
        return redirect(request.url)

    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Create a safe filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Convert the image to black and white before saving
    try:
        image = Image.open(file)
        exif_data = image._getexif()

        if exif_data is not None:
            # Correct the orientation of the image
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(exif_data.items())

            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)

        image = image.convert('L')
        image.save(filepath)
    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(request.url)

    # Colorize the image
    colorized_img_path = colorize_image(filepath)
    
    return render_template('index.html', input_image=filename, output_image=os.path.basename(colorized_img_path))
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def colorized_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)
def boost_blue(img_bgr):
    """
    Boost the blue component in the image where blue is predominant.
    """
    # Set a threshold for detecting blue predominant regions.
    blue_threshold = 100
    blue_boost_factor = 1.2
    
    # Create a mask where blue is the predominant channel.
    blue_mask = (img_bgr[:,:,0] > blue_threshold) & (img_bgr[:,:,0] > img_bgr[:,:,1]) & (img_bgr[:,:,0] > img_bgr[:,:,2])
    
    # Boost the blue channel in the regions detected.
    img_bgr[blue_mask, 0] = np.clip(img_bgr[blue_mask, 0] * blue_boost_factor, 0, 255)
    
    return img_bgr
def colorize_image(filepath):
    # Read the image
    img = cv2.imread(filepath, 1)
    img_rgb = img.astype(np.float32) / 255.
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0] # Extract the L channel

    # Resize the L channel to network input size
    img_l_rs = cv2.resize(img_l, (224, 224))
    img_l_rs -= 50 # Mean centering

    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize back to original size
    ab_dec_us = cv2.resize(ab_dec, (img.shape[1], img.shape[0]))
    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)

    # Convert LAB to RGB
    img_bgr_out = cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR)
    img_bgr_out = np.clip(img_bgr_out * 255, 0, 255).astype(np.uint8)

    # Save the colorized image
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], 'colorized_' + os.path.basename(filepath))
    cv2.imwrite(output_filepath, img_bgr_out)
    
    return output_filepath

if __name__ == '__main__':
    app.run(debug=True)
