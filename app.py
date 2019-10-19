import os
from flask import Flask, request, jsonify
from convert.keras import KerasONNX
import onnxruntime as rt
import numpy as np
from PIL import Image
import requests
from google.cloud import storage
from io import BytesIO, StringIO


from config import bucketName

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/convert/<project_id>', methods = ['POST'])
def keras_to_onnx(project_id):
    # TODO: Get model information from project id @model_path and @save_path
    model_path = request.get_json()['model_path']

    success = False 

    try:
        keras_onnx = KerasONNX(project_id + "/" + model_path)
        keras_onnx.convert()
        keras_onnx.save()
        success = True
    except:
        success = False

    return jsonify(success=success)


@app.route('/inference/<project_id>', methods=['POST'])
def image_classification_inference(project_id):
    # TODO: Get model information from project id @onnx_model
    # ONNX Model
    model_path = request.get_json()['model_path']
    image_url = request.get_json()['image_url']

    model_path = project_id + "/" + model_path

    if not os.path.isfile("tmp/"+model_path):
        gcs = storage.Client()
        bucket = gcs.get_bucket(bucketName)
        blob = bucket.blob(model_path)
        blob.download_to_filename("tmp/" + model_path)

    sess = rt.InferenceSession("tmp/" + model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    expected_shape = sess.get_inputs()[0].shape

    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    response.raw.decode_content = True
    image = preprocess(response.raw, shape=(100, 100, 1), grayscale=True, normalize=True)

    result = sess.run([], {input_name: image})

    # TODO: Remap the classes
    classes = ["car", "not a car"]
    prob = result[0][0]
    
    return jsonify(success=True, result=classes[np.argmax(prob)], confidence=str(prob.max()))


def preprocess(image_url, shape=(224, 224, 3), grayscale=False, normalize=True):
    image = Image.open(image_url)
    if (grayscale):
        image = image.convert('L')

    image = image.resize(shape[:-1])
    image = np.array(image)

    if (normalize):
        image = image.astype(np.float32) / 255

    image = np.array([image])
    image = np.array([image]).reshape(shape)
    image = image[np.newaxis]

    return image


@app.route('/upload/<project_id>', methods=['POST'])
def upload(project_id):
    # TODO: get all information about the project using project_id
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return 'No file uploaded.', 400

    gcs = storage.Client()

    bucket = gcs.get_bucket(bucketName)

    blob = bucket.blob(project_id + "/" + uploaded_file.filename)

    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )

    return blob.public_url


if __name__ == '__main__':
    app.run(debug=True)
