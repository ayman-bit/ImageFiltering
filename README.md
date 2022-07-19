# ImageFiltering

The Laplacian filter is used in image processing applications as an edge detector. It is a two-dimensional approximation of the second spatial derivative of an image. However, the Laplacian filter is known to amplify noise, and so often the image is first blurred with a Gaussian filter to reduce noise levels so that amplification with the Laplacian filter does not become a major issue.
Instead of first applying the Gaussian filter followed by application of the Laplacian filter, it is possible to combine these two steps into one and apply a single Laplacian-of-Gaussian filter.

For the purposes of this assignment, we will assume the following discrete kernel with a value of 1.4.

The actual image filtering/convolution should be implemented using the Message Passing Interface (MPI) with each node responsible for calculating the output for a single 64x64 subimage in the same PGM format as the input.

Note that near the boundaries of each subimage, pixel values from outside that subimage will be required to calcualte the filter output. As a result, it will be necessary to send messages from the node responsible for one subimage to the nodes responsible for neighboring subimages in order to properly calculate the filter output. You may also assume that outside of the boundary of the entire image, all values are equal to zero (this will affect the quality of the filtering at the image boundaries, but simplifies the overall implementation).

The output of the entire MPI program should be a file containing the data in the same PGM format as the given input 256x256 image which is the result of filtering the image with the above kernel.
