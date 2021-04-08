# import tensorflow as tf
# model = tf.keras.models.load_model('pspunet_weight.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

from tensorflow import keras
model = keras.models.load_model('pspunet_weight.h5', compile=False)
 
export_path = '/Users/kim-yulhee/Smart-Cane-using-semantic-segmentation'
model.save(export_path, save_format="tf")

