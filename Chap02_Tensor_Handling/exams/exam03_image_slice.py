'''
문) image.jpg 이미지 파일을 대상으로 파랑색 우산 부분만 slice 하시오.
'''

import matplotlib.image as img
import matplotlib.pyplot as plt
import tensorflow as tf

filename = 'C:/ITWILL/6_Tensorflow/data/image.jpg'
input_image = img.imread(filename)

output_image = tf.slice(input_image, [105,30,0],[-1,540,-1])
plt.subplot(211)
plt.imshow(input_image)
plt.subplot(212)
plt.imshow(output_image)
plt.show()