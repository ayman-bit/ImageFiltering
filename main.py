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

img = cv2.imread('butterfly.jpg', 0)

# Logic of convolution between the kernel and the image


cv2.imshow('butterfly', img)
cv2.waitKey(10000)

