import cv2
import tensorflow as tf
from data_loader.display import create_mask, show_predictions
import numpy as np
from PIL import Image
import pandas as pd

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

interpreter = tf.lite.Interpreter(tensorflow_lite_model_file)
# Load TFLite model and allocate tensors.

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# input details
print("----------input details----------")
print(input_details[0]['index'])

img = cv2.imread('./surface_img/data1.jpeg')
img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img / 255

img = tf.expand_dims(img, 0)

input_data = np.array(img, dtype=np.float32)
#print(input_data.shape)
'''
Get indexes of input and output layers
input_details[0]['index']를 출력하면 0, 딕셔너리 안에 index라는 key가 있고 그 Index값이 0임.
[1,IMG_HEIGHT, IMG_WIDTH,3] -> input에 들어가는 image의 shape 형태 [갯수, height, width, 채널]
'''
interpreter.resize_tensor_input(input_details[0]['index'],[1, IMG_HEIGHT, IMG_WIDTH, 3])
# allocate_tensor
interpreter.allocate_tensors()
'''
Transform input data (tensor_index, value)
tensor_index: Tensor index of tensor to set. This value can be gotten from the 'index' field in get_input_details.
value:	Value of tensor to set.
'''
interpreter.set_tensor(input_details[0]['index'], input_data)
# run the inference
interpreter.invoke()
# output_details[0]['index'] = the index which provides the input
output_data = interpreter.get_tensor(output_details[0]['index'])

pre = create_mask(output_data).numpy()

#result1 = np.array(pre).T[0][0]
result1 = np.array(pre).T[0][0:80]
result2 = np.array(pre).T[0][80:160]
result3 = np.array(pre).T[0][160:240]
result4 = np.array(pre).T[0][240:320]
result5 = np.array(pre).T[0][320:400]
result6 = np.array(pre).T[0][400:480]

print(result1)
print("=================================")
print(result2)
caution1_2 = []
caution1_3 = []
caution1_5 = []
for j in result1:
    for x in j:
        if x == 3:
            caution1_3.append(j)
        elif x == 2:
            caution1_2.append(j)
        elif x == 5:
            caution1_5.append(j)

print("label2: ", len(caution1_2))
print("label3: ", len(caution1_3))
print("label5: ", len(caution1_5))

caution2_2 = []
caution2_3 = []
caution2_5 = []
for j in result2:
    for x in j:
        if x == 3:
            caution2_3.append(j)
        elif x == 2:
            caution2_2.append(j)
        elif x == 5:
            caution2_5.append(j)

print("label2: ", len(caution2_2))
print("label3: ", len(caution2_3))
print("label5: ", len(caution2_5))

caution3_2 = []
caution3_3 = []
caution3_5 = []
for j in result2:
    for x in j:
        if x == 3:
            caution3_3.append(j)
        elif x == 2:
            caution3_2.append(j)
        elif x == 5:
            caution3_5.append(j)

print("label2: ", len(caution3_2))
print("label2: ", len(caution3_3))
print("label5: ", len(caution3_5))

caution4_2 = []
caution4_3 = []
caution4_5 = []
for j in result2:
    for x in j:
        if x == 3:
            caution4_3.append(j)
        elif x == 2:
            caution4_2.append(j)
        elif x == 5:
            caution4_5.append(j)

print("label2: ", len(caution4_2))
print("label2: ", len(caution4_3))
print("label5: ", len(caution4_5))

caution5_2 = []
caution5_3 = []
caution5_5 = []
for j in result2:
    for x in j:
        if x == 3:
            caution5_3.append(j)
        elif x == 2:
            caution5_2.append(j)
        elif x == 5:
            caution5_5.append(j)

print("label2: ", len(caution5_2))
print("label2: ", len(caution5_3))
print("label5: ", len(caution5_5))

caution6_2 = []
caution6_3 = []
caution6_5 = []
for j in result2:
    for x in j:
        if x == 3:
            caution6_3.append(j)
        elif x == 2:
            caution6_2.append(j)
        elif x == 5:
            caution6_5.append(j)

print("label2: ", len(caution6_2))
print("label2: ", len(caution6_3))
print("label5: ", len(caution6_5))