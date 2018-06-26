import numpy as np
from keras.models import load_model
from PIL import Image
import PIL.ImageOps
import tensorflow as tf


graph = tf.get_default_graph()
model = load_model("../model.h5")
# 加载文字标签文件
path = 'chinese.txt'
lie = []
label = []
# 初始化lie列表
for line in open(path):  # 遍历txt文件中的所有行
        line = line.replace('\n', '').split(",")  # 替换和分割
        lie.append(line)  # 将第六行的数据重新存在lie中
lie[0][0] = '一'
label = lie[0]


def img2class(image_paths):
    images = []
    for f_name in image_paths:
        img = Image.open(f_name).convert('RGB')  # 打开图片，并将图片改为ＲＧＢ格式
        img = PIL.ImageOps.invert(img)  # 将图片进行反转使黑底变为白底
        img = np.array(img).astype(np.float32) / 255.  # 对图片进行预处理
        images.append(img)
    x = np.array(images)
    global graph  # “投放”到session中的图s
    with graph.as_default():
        result = model.predict(ｘ)
        print('result:', result)
        result = np.argmax(result[0])
        result = label[result]  # 将数字标签改为文字标签
    return result
