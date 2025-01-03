# NLA - Challenge 2

**Goal**: apply the singular value decomposition to image compression and noise reduction.

**Data**: download the file [einstein.jpg][1] and move it to your working directory.

**Tasks**:
1. Load the image as an Eigen matrix with size $m \times n$.
   Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 
   (black) and 255 (white).
   Compute the matrix product $A^{T}A$ and report the Euclidean norm of $A^{T}A$.
   
   Answer: $1050410679.542489$
2. Solve the eigenvalue problem $A^{T}Ax = \lambda x$ using the proper solver provided by the Eigen library.
   Report the two largest computed singular values of A.

   Answer: $32339.103442$ (largest), $9523.101669$ (second largest).
3. Export matrix $A^{T}A$ in the matrix market format and move it to the `lis-2.1.6/test` folder.
   Using the proper iterative solver available in the LIS library compute the largest eigenvalue of $A^{T}A$
   up to a tolerance of $10^{-8}$. Report the computed eigenvalue. 
   Is the result in agreement with the one obtained in the previous point?

   Answer: yes, the value obtained is $1.045818e+09$
4. Find a shift $\mu \in \mathbb{R}$ yielding an acceleration of the previous eigensolver.
   Report $\mu$ and the number of iterations required to achieve a tolerance of $10^{-8}$.

   Answer: $\mu = 4.244525\text{e+}07$ and $7$ iterations.
5. Using the SVD module of the Eigen library, perform a singular value decomposition of the
   matrix $A$. Report the Euclidean norm of the diagonal matrix $\Sigma$ of the singular values.

   Answer: $35576.621650$
6. Compute the matrices $C$ and $D$ assuming $k = 40$ and $k = 80$.
   Report the number of nonzero entries in the matrices $C$ and $D$.
   
   The truncated SVD considering the first $k$ terms, with $k < r$ is the best approximation of the matrix $A$
   among the matrices of the rank at most $k$ in the sense of Frobenius norm.
   This can be used for image compression as follows: instead of storing the whole $m \times n$ matrix $A$,
   we can instead store the $m \times k$ and $n \times k$ matrices
   $C = [\mathbf{u}_{1} \: \mathbf{u}_{2} \: \dots \: \mathbf{u}_{k}]$,
   $D = [\sigma_{1}\mathbf{v}_{1} \: \sigma_{2}\mathbf{v}_{2} \: \dots \: \sigma_{r}\mathbf{v}_{k}]$.
   If $k$ is much smaller than $p$, then storing $C$ and $D$ will take much less space than storing the full matrix $A$.
   The compressed image can be simply computed as $\tilde{A} = CD^{T}$.

   Answer: ($k = 40$, $\mathrm{nnz}(C) = 13640$, $\mathrm{nnz}(D) = 10240$), 
   ($k = 80$, $\mathrm{nnz}(C) = 27280$, $\mathrm{nnz}(D) = 20480$).
7. Compute the compressed images as the matrix product $CD^{T}$ (again for $k = 40$ and $k = 80$).
   Export and upload the resulting images in `.png`.

   Answer: see the pictures [compressed_k40](resources/compressed_image_k40.png) and
   [compressed_k80](resources/compressed_image_k80.png).

   <img alt="compressed img k40" src="resources/compressed_image_k40.png">
   <img alt="compressed img k80" src="resources/compressed_image_k80.png">
8. Using `Eigen` create a black and white checkerboard image with height and width equal to 200 pixels.
   Report the Euclidean norm of the matrix corresponding to the image.

   Answer: $36062.445841$
9. Introduce a noise into the checkerboard image by adding random fluctuations 
   of color ranging between $[-50, 50]$ to each pixel.
   Export the resulting image in `.png` and upload it.

   Answer: see the figure [noise.png](resources/noise.png).
   
   <img alt="noise image" src="resources/noise.png">
10. Using the SVD module of the Eigen library, 
    perform a singular value decomposition of the matrix corresponding to the noisy image.
    Report the two largest computed singular values.
    
    Answer: $25495.147715$ (largest), $22946.704097$ (second largest).
11. Starting from the previously computed SVD, creates the matrices C and D defined in (1)
    assuming $k = 5$ and $k = 10$. Report the size of the matrices $C$ and $D$.

    Answer: (k = 5, C = (200 rows, 5 cols), D = (200 rows, 5 cols)), 
    (k = 10, C = (200 rows, 10 cols), D = (200 rows, 10 cols)).
12. Compute the compressed images as the matrix product $CD^{T}$ (again for $k = 5$ and $k = 10$).
    Export and upload the resulting images in `.png`.

    Answer: see the pictures [compressed_k5](resources/compressed_noise_image_k5.png) and
    [compressed_k10](resources/compressed_noise_image_k10.png).

    <img alt="compressed noise image k5" src="resources/compressed_noise_image_k5.png">
    <img alt="compressed noise image k10" src="resources/compressed_noise_image_k10.png">
13. Compare the compressed images with the original and noisy images. Comment the results.

    Answer: A low $k$ value means higher compression because fewer terms are used.
    This is clearly seen if we look at the image of Einstein with $k = 40$ compared to the image with $k = 80$; 
    the first image has lost a lot of details and the image is not of good quality;
    in contrast, the second image has more details because it contains more values ($k$ is larger). 
    Another interesting result is that a low value of $k$ also reduces the noise. 
    This can be seen in the chessboard image, where the $k = 5$ image has less noise than the $k = 10$ image. 
    So what do these tests tell us? 
    Well, that the truncated svd method needs to be used carefully so that we can use it to balance 
    data compression and detail preservation. 
    Obviously, we can't use $k$ too high (+ details, noise doesn't change much) or $k$ too low 
    (+ compression, reduces noise, - details). 
    However, low $k$-values work well (in terms of noise reduction) for images with little detail, 
    such as a chessboard image, but $k$-values reduce the details 
    too much for an image with a lot of detail (Einstein image).

[1]: https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/256px-Albert_Einstein_Head.jpg?20141125195928=&download=
