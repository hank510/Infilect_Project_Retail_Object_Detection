import os

import cv2
import numpy as np
from io import BytesIO
from flask import Flask, request, render_template, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired

from inference import process_image_inference  # Import inference function
from grouping import group_similar_images  # Import grouping function
from create_diagram import create_labeled_image  # Import create diagram function

app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'
app.config['UPLOAD_FOLDER'] = 'static/images'

# Define the form for file upload
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload and Process File")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    
    # If the form is submitted and validated
    if form.validate_on_submit():
        # Step 1: Receive and load image into memory
        file = form.file.data
        in_memory_image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(in_memory_image, cv2.IMREAD_COLOR)

        # Step 2: Process image through inference.py to get bounding boxes
        bounding_boxes = process_image_inference(image)

        # Step 3: Use bounding box data in grouping.py to group similar images
        groups = group_similar_images(image, bounding_boxes)

        # Step 4: Save the image with bounding boxes and labels
        labeled_image = create_labeled_image(image, groups)
        labeled_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'labeled_image.jpg')
        cv2.imwrite(labeled_image_path, labeled_image)

        # Pass the image URL and groups to the template
        return render_template('index.html', groups=groups, labeled_image_url=url_for('static', filename='images/labeled_image.jpg'), form=form)

        # # Return the final groups as JSON response in the rendered page
        # return groups, labeled_image

    # Render the HTML template with the form
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
