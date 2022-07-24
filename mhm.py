import numpy as np
import cv2
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
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
image = cv2.imread(filepath, 0)

M = 64
N = 64

if rank == 0:
    data = [
        image[x : x + M, y : y + N]
        for x in range(0, image.shape[0], M)
        for y in range(0, image.shape[1], N)
    ]
else:
    data = None

data = comm.scatter(data, root=0)
data = convolve(data, kernel)
print("rank", rank, "has data:", data)

convolvedImage = comm.gather(data, root=0)

if rank == 0:
    cv2.imshow("image", combineTilesIntoOneImage(convolvedImage))
    cv2.waitKey(5000)


def combineTilesIntoOneImage(convolvedImage):
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

    return cv2.vconcat([im_v, im_v2, im_v3, im_v4])
