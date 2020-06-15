'''
 image slice
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

# image slice
img_slice = tf.slice(input_image, [15,0,0],[16,-1,-1])

print(img_slice.shape) 

# image slice 출력 
plt.imshow(img_slice)
plt.show()

