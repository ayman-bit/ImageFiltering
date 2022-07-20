import numpy as np
import cv2
from mpi4py import MPI

kernel = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                   [0, 2, 3, 5, 5, 5, 3, 2, 0],
                   [3, 3, 5, 3, 0, 3, 5, 3, 3],
                   [2, 5, 3, -12, -23, -12, 3, 5, 2],
                   [2, 5, 0, -23, -40, -23, 0, 5, 2],
                   [2, 5, 3, -12, -23, -12, 3, 5, 2],
                   [3, 3, 5, 3, 0, 3, 5, 3, 3],
                   [0, 2, 3, 5, 5, 5, 3, 2, 0],
                   [0, 0, 3, 2, 2, 2, 3, 0, 0]])

filepath = 'pepper.ascii.pgm'

image = cv2.imread(filepath,0)

# Convolute the mask with the image. May only work for masks of odd dimensions
convolvedImage = cv2.filter2D(image,-1,kernel)

cv2.imshow('image', convolvedImage)
cv2.waitKey(10000)
cv2.imwrite('answer.ascii.pgm',convolvedImage)