# NLA - Challenge 1

**Goal**: apply image filters and find the approximate solution of linear system to process a greyscale image.

**Data**: download the file [einstein.jpg](https://commons.wikimedia.org/wiki/File:Albert_Einstein_Head.jpg)
([256px][1])
and move it to your working directory.

**Tasks**:
1. Load the image as an Eigen matrix with size $m \times n$. 
   Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere 
   between 0 (black) and 255 (white). Report the size of the matrix.
   
   Answer: $341 \times 256$ ($rows \times cols$).

2. Introduce a noise signal into the loaded image by adding random fluctuations of color ranging 
   between $[-50, 50]$ to each pixel. Export the resulting image in `.png` and upload it.

   Answer: see the figure [noise.png](resources/noise.png).

   <img src="resources/noise.png">

3. Reshape the original and noisy images as vectors $v$ and $w$, respectively. 
   Verify that each vector has $m \: n$ components. Report here the Euclidean norm of $v$.

   Answer: $35576.621650$

4. Write the convolution operation corresponding to the smoothing kernel $H_{av2}$ as a matrix vector multiplication 
   between a matrix $A_{1}$ having size $mn \times mn$ and the image vector.

   Report the number of non-zero entries in $A_{1}$.

```math
 H_{av2} = \dfrac{1}{9}\begin{bmatrix}
     1 & 1 & 1 \\
     1 & 1 & 1 \\
     1 & 1 & 1
 \end{bmatrix}
```


[1]: https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/256px-Albert_Einstein_Head.jpg?20141125195928=&download=
