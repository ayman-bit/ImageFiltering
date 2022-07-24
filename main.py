import numpy as np
import cv2
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def convolve(img, kernel):
    img_w = img.shape[1]
    img_h = img.shape[0]

    k_size = kernel.shape[0]

    img2_w = img_w - k_size + 1
    img2_h = img_h - k_size + 1

    conv = np.zeros((img2_h, img2_w), "float")

    for x in range(img2_h):
        for y in range(img2_w):
            conv[x][y] = float(
                np.sum(np.multiply(kernel, img[x : x + k_size, y : y + k_size]))
            )
    return conv


kernel = np.array(
    [
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [2, 5, 0, -23, -40, -23, 0, 5, 2],
        [2, 5, 3, -12, -23, -12, 3, 5, 2],
        [3, 3, 5, 3, 0, 3, 5, 3, 3],
        [0, 2, 3, 5, 5, 5, 3, 2, 0],
        [0, 0, 3, 2, 2, 2, 3, 0, 0],
    ]
)

filepath = "pepper.ascii.pgm"

# if rank == 0:
#     # in real code, this section might
#     # read in data parameters from a file
#     numData = 10
#     comm.send(numData, dest=1)

#     data = np.linspace(0.0,3.14,numData)
#     comm.Send(data, dest=1)

# elif rank == 1:

#     numData = comm.recv(source=0)
#     print('Number of data to receive: ',numData)

#     data = np.empty(numData, dtype='d')  # allocate space to receive the array
#     comm.Recv(data, source=0)

#     print('data received: ',data)

image = cv2.imread(filepath, 0)
# cv2.imshow('image', image)
# cv2.waitKey(10000)
# Convolute the mask with the image. May only work for masks of odd dimensions
# convolvedImage = cv2.filter2D(image,-1,kernel)

# cv2.imshow('image', convolvedImage)
# cv2.waitKey(10000)
# cv2.imwrite('answer.ascii.pgm',convolvedImage)

M = 64
N = 64
tiles = [
    image[x : x + M, y : y + N]
    for x in range(0, image.shape[0], M)
    for y in range(0, image.shape[1], N)
]
convolvedImage = []

for images in tiles:
    convolvedImage.append(convolve(images, kernel))


im_v = cv2.hconcat(
    [convolvedImage[0], convolvedImage[1], convolvedImage[2], convolvedImage[3]]
)
im_v2 = cv2.hconcat(
    [convolvedImage[4], convolvedImage[5], convolvedImage[6], convolvedImage[7]]
)
im_v3 = cv2.hconcat(
    [convolvedImage[8], convolvedImage[9], convolvedImage[10], convolvedImage[11]]
)
im_v4 = cv2.hconcat(
    [convolvedImage[12], convolvedImage[13], convolvedImage[14], convolvedImage[15]]
)

im_v_final = cv2.vconcat([im_v, im_v2, im_v3, im_v4])
cv2.imshow("image", im_v_final)
cv2.waitKey(1000)


# cv2.imshow('image2', fullImage)
# cv2.imwrite('answer.ascii.pgm', fullImage)
cv2.waitKey(10000)
