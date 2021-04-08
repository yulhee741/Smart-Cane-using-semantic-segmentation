import tensorflow as tf
 
saved_model_dir = '/Users/kim-yulhee/Smart-Cane-using-semantic-segmentation'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('/Users/kim-yulhee/Smart-Cane-using-semantic-segmentation/converted_model.tflite', 'wb').write(tflite_model)

