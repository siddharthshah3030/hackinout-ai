from flask import Flask, request, jsonify
from convert.keras import KerasONNX
import onnxruntime as rt
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/convert/<project_id>', methods = ['POST'])
def keras_to_onnx(project_id):
    # TODO: Get model information from project id @model_path and @save_path
    model_path = request.get_json()['model_path']
    save_path = request.get_json()['save_path']

    success = False 

    try:
        keras_onnx = KerasONNX(model_path, save_path)
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

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    expected_shape = sess.get_inputs()[0].shape

    response = requests.get(image_url, stream=True)
    response.raw.decode_content = True
    image = preprocess(response.raw, shape=(100, 100, 1), grayscale=True, normalize=True)

    print(image.shape)

    result = sess.run([], {input_name: image})

    # TODO: Remap the classes
    classes = ["car", "not a car"]
    prob = result[0][0]
    
    return jsonify(success=True, result=classes[np.argmax(prob)], confidence=str(prob.max()))


def preprocess(image_url, shape=(224, 224, 3), grayscale=False, normalize=True):
    image = Image.open(image_url)
    if (grayscale):
        image = image.convert('L')

    print(shape)
    image = image.resize(shape[:-1])
    image = np.array(image)

    if (normalize):
        image = image.astype(np.float32) / 255

    image = np.array([image])
    image = np.array([image]).reshape(shape)
    image = image[np.newaxis]

    return image

if __name__ == '__main__':
    app.run(debug=True)
