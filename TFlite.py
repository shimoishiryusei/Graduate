import numpy as np
import tensorflow as tf
import cv2
import time

# TensorFlow Liteモデルをロードする
interpreter = tf.lite.Interpreter(model_path='TFlite_model.tflite')

# TensorFlow Liteモデルを実行する
interpreter.allocate_tensors()


# 画像を読み込んで前処理を行う
image = cv2.imread("11ULT_20002000.jpg")
image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# 入力データを準備する
input_tensor_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_tensor_index, image)

start_time = time.time()
# TensorFlow Liteを使用して予測を行う
interpreter.invoke()

end_time = time.time()
elapsed_time = end_time - start_time

# 結果を取得する
output_tensor_index = interpreter.get_output_details()[0]["index"]
output_data = interpreter.get_tensor(output_tensor_index)

# 結果を表示する
print(output_data)
print("elapsed_time:", elapsed_time)