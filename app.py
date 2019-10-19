from flask import Flask
from convert.keras import KerasONNX

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


def keras_to_onnx(model_path, save_path):
    keras_onnx = KerasONNX(model_path, save_path)
    keras_onnx.convert()
    keras_onnx.save()


if __name__ == '__main__':
    keras_to_onnx('mobilenet.h5', 'mobilenet.onnx')
    app.run(debug=True)
