from flask import Flask, request, jsonify
from convert.keras import KerasONNX

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


if __name__ == '__main__':
    app.run(debug=True)
