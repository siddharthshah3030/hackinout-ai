import onnxmltools
import tensorflow as tf
from keras.models import load_model
from keras import backend as K


class KerasONNX():
    def __init__(self, model_path, onnx_model_path):
        K.clear_session()
        self.model = tf.reset_default_graph()

        self.model_path = model_path
        self.onnx_model_path = onnx_model_path
        self.model = load_model(model_path)

    def convert(self):
        self.onnx = onnxmltools.convert_keras(self.model)

    def save(self):
        onnxmltools.utils.save_model(self.onnx, self.onnx_model_path)