import KerasONNX

def keras_to_onnx(model_path, save_path):
    keras_onnx = KerasONNX(model_path, save_path)
    keras_onnx.convert()
    keras_onnx.save()

keras_to_onnx('mobilenet.h5', 'mobilenet.onnx')