import tensorflow as tf
from PIL import Image
import cv2

# Load the model.
model = tf.keras.models.load_model('colorizer.h5')

# Capture a video stream.
cap = cv2.VideoCapture(0)
link =" "
# Loop until the user presses the Esc key.
while True:
   
    # Capture a frame from the video stream.
    ret, frame = cap.read(link)

    # Convert the frame to a NumPy array.
    frame = tf.keras.preprocessing.image.img_to_array(frame)

    # Colorize the frame.
    colored_frame = model.predict(frame)[0]

    # Convert the NumPy array back to an image.
    colored_frame = tf.keras.preprocessing.image.array_to_img(colored_frame)

    # Display the colored frame.
    cv2.imshow('Colorized Frame', colored_frame)

    # Wait for a key press.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture resource.
cap.release()

# Destroy all windows.
cv2.destroyAllWindows()