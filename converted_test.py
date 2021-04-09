import cv2
import tensorflow as tf
from data_loader.display import create_mask
import numpy as np
from PIL import Image
import tflite-runtime.interpreter as tflite

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

IMG_WIDTH = 480
IMG_HEIGHT = 272
n_classes = 7


tensorflow_lite_model_file = "/Users/kim-yulhee/Smart-Cane-using-semantic-segmentation/converted_model.tflite"
#my_signature = interpreter.get_signature_runner()
interpreter.allocate_tensors()

interpreter = tflite.Interpreter(tensorflow_lite_model_file)
img = cv2.imread('./surface_img/data1.jpeg')
img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img / 255

img = tf.expand_dims(img, 0)

input_data = img

interpreter.set_tensor(input_data)

interpreter.invoke()

output_data = sep_interpreter.get_tensor()

pre = create_mask(output_data).numpy()


print(pre)

frame2 = img/2
print("===============")


#frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
#cv2.imwrite('./output/result.png', img1)