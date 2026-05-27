import cv2
import os
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename

# ------------------------------
# Flask App Configuration
# ------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Webcam Initialization
camera = cv2.VideoCapture(0)

# ------------------------------
# Helper Functions
# ------------------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)

    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    inverted_blur = cv2.bitwise_not(blurred)

    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    return sketch


# ------------------------------
# Routes
# ------------------------------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/sketch', methods=['POST'])
def sketch():

    if 'file' not in request.files:
        return render_template('home.html', error="No file selected")

    file = request.files['file']

    if file.filename == '':
        return render_template('home.html', error="Please select an image")

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        # Read Image
        img = cv2.imread(filepath)

        # Generate Sketch
        sketch_img = make_sketch(img)

        # Save Sketch Image
        sketch_name = filename.rsplit('.', 1)[0] + "_sketch.jpg"

        sketch_path = os.path.join(app.config['UPLOAD_FOLDER'], sketch_name)

        cv2.imwrite(sketch_path, sketch_img)

        return render_template(
            'home.html',
            org_img_name=filename,
            sketch_img_name=sketch_name
        )

    return render_template('home.html', error="Invalid File Format")


# ------------------------------
# Live Camera Sketch Feed
# ------------------------------

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        else:

            sketch_frame = make_sketch(frame)

            ret, buffer = cv2.imencode('.jpg', sketch_frame)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame + b'\r\n')


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ------------------------------
# Run App
# ------------------------------

if __name__ == '__main__':
    app.run(debug=True)