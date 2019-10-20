import tensorflow as tf

def load_image(image_path):
    image_path = image_path
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    return image