import numpy as np
import cv2
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of nodes, in this case 16
rank = comm.Get_rank()  # gets the current rank

# This method takes an image and kernel, and output the convolved image
def convolve(img, kernel):
    #finds the image dimensions
    img_w = img.shape[1]
    img_h = img.shape[0]
    #finds the kernel size
    kernel_size = kernel.shape[0]

    #Obtains the convolved height and weight from the kernel and source image
    convo_w = (img_w - kernel_size) + 1
    convo_h = (img_h - kernel_size) + 1

    convolved = np.zeros((convo_h, convo_w), "float")

    #for-loop that convolutes the image and kernel
    for x in range(convo_h):
        for y in range(convo_w):
            convolved[x][y] = float(
                np.sum(np.multiply(kernel, img[x : x + kernel_size, y : y + kernel_size]))
            )
    return convolved


# This method takes an array of the 16 convoluted 64x64 tiles, and combines them to the final image
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


# given kernel matrix
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

# read the input file
filepath = "pepper.ascii.pgm"
image = cv2.imread(filepath, 0)

M = 64
N = 64

if rank == 0:
    # divide the input image into 16 tiles of 64x64
    data = [
        image[x : x + M, y : y + N]
        for x in range(0, image.shape[0], M)
        for y in range(0, image.shape[1], N)
    ]
else:
    data = None

# MPI_SCATTER to scatter the 16 tiles among the 16 nodes
data = comm.scatter(data, root=0)

# Convolve the tile using the given kernel
data = convolve(data, kernel)
print("rank", rank, "has data:", data)

# MPI_GATHER to gather all 16 individual convolved tiles into one array
convolvedImage = comm.gather(data, root=0)

if rank == 0:
    # combine the tiles and show the image for 5 seconds
    final_output = combineTilesIntoOneImage(convolvedImage)
    cv2.imshow("image", final_output)
    cv2.waitKey(5000)

    # write the final output into a png file
    cv2.imwrite("final_output.png", final_output)
