import cv2
import tensorflow as tf
from model.pspunet import pspunet
from model.pspunet import pspunet
from data_loader.display import create_mask
import numpy as np

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

model = pspunet((IMG_HEIGHT, IMG_WIDTH, 3), n_classes)
model.load_weights("pspunet_weight.h5")

img = cv2.imread('./surface_img/data1.jpg')
img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = tf.expand_dims(img, 0)
pre = model.predict(img)
pre = create_mask(pre).numpy()

print(pre)