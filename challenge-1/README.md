# NLA - Challenge 1

**Goal**: apply image filters and find the approximate solution of linear system to process a greyscale image.

**Data**: download the file [einstein.jpg](https://commons.wikimedia.org/wiki/File:Albert_Einstein_Head.jpg) (256px)
and move it to your working directory.

**Tasks**:
1. Load the image as an Eigen matrix with size $m \times n$. 
   Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere 
   between 0 (black) and 255 (white). Report the size of the matrix.
2. Introduce a noise signal into the loaded image by adding random fluctuations of color ranging 
   between $[-50, 50]$ to each pixel. Export the resulting image in `.png` and upload it.