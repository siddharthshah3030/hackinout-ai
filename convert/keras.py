import os
import onnxmltools
from keras.models import load_model
import keras2onnx
from tensorflow.python.lib.io import file_io
from google.cloud import storage


from config import bucketName


class KerasONNX():
    def __init__(self, model_path):
        self.model_path = model_path
        project_id = self.model_path.split('/')[0]

        if not os.path.exists("tmp/" + project_id):
            os.makedirs("tmp/" + project_id)

        self.temp_model_location = os.path.join('tmp', self.model_path)
        self.save_temp()

        self.model = load_model(self.temp_model_location)

    def convert(self):
        try:
            self.onnx = onnxmltools.convert_keras(self.model)
            return True
        except:
            return False
            

    def save(self):
        try:
            onnxmltools.utils.save_model(self.onnx, self.temp_model_location + ".onnx")
            self.save_onnx()
            return True
        except:
            return False

    def save_onnx(self):
        try:
            gcs = storage.Client()
            bucket = gcs.get_bucket(bucketName)

            blob = bucket.blob(self.model_path + ".onnx")

            blob.upload_from_filename(
                self.temp_model_location + ".onnx"
            )
        except:
            return False

        return True



    def save_temp(self):
        gcs = storage.Client()
        bucket = gcs.get_bucket(bucketName)
        blob = bucket.blob(self.model_path)
        blob.download_to_filename(self.temp_model_location)
