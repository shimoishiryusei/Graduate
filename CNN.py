from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
model = load_model('CNNmodel.h5')    

from skimage import io
import numpy as np
from skimage.transform import resize
import tensorflow as tf

img = io.imread('./11ULT_30001000.jpg')
io.imshow(img)
pred = model.predict(np.expand_dims(img[:,:,0:3]/255, axis=0))
print(pred)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("TFlite_model.tflite", "wb").write(tflite_model)