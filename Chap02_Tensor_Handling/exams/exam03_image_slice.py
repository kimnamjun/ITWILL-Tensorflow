'''
문) image.jpg 이미지 파일을 대상으로 파랑색 우산 부분만 slice 하시오.
'''

import matplotlib.image as mp_image
filename = "../data/image.jpg"
input_image = mp_image.imread(filename)
