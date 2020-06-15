'''
image 변환
'''

import matplotlib.image as img 
import matplotlib.pyplot as plt 
import tensorflow as tf

filename = 'C:/ITWILL/6_Tensorflow/data/packt.jpeg'
input_image = img.imread(filename)

print('input dim =', input_image.ndim) #dimension
print('input shape =', input_image.shape) #shape

# image 원본 출력
plt.imshow(input_image)
plt.show()

# image transpose : 축 변경
img_tran = tf.transpose(a=input_image, perm=[1, 0, 2])
plt.imshow(img_tran)
plt.show()
