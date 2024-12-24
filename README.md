# Numerical Linear Algebra challenges

## Description

The repository contains a set of challenges to be solved using knowledge of numerical linear algebra and the famous linear algebra library: [Eigen][Eigen]. In addition, the use of [LIS (Library of Iterative Solvers for Linear Systems)][LIS] has been adopted to solve some eigenvalue problems. The two challenges are developed using the C++ programming language.

The challenges are part of the [Numerical Linear Algebra][NLA] course at the [Politecnico di Milano][POLIMI].

The challenges are divided into two main categories:
- [Challenge 1](challenge-1/README.md). Apply image filters and find the approximate solution of a linear system to process a grayscale image.
- [Challenge 2](challenge-2/README.md). Apply [Singular Value Decomposition (SVD)][SVD] for image compression and noise reduction.

Each folder contains a README that explains each challenge in more detail.

---------------------------------------------------------------------

## How to run the code

Eigen and LIS libraries are required to run the code. The code has been tested on Linux OS, therefore we don't guarantee that it will run correctly.
- Eigen doesn't have any dependencies other than the C++ standard library. It is a pure template library defined in the headers. The [repository][Eigen-Repo] contains all the releases.
- Installing Lis requires a C compiler. More information can be found on the [official website][LIS].

If you are a student at the [Politecnico di Milano][POLIMI], you can easily use the [MK Library][MK] (provided by the [MOX Laboratory][MOX]) to compile the code.

In the [CMakeLists.txt file](./CMakeLists.txt), you can find the following lines that include the [MK library][MK]:
```cmake
include_directories(
        /u/sw/toolchains/gcc-glibc/11.2.0/base/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/lis/2.0.30/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/include
)
```

If you don't want to use the MK library, you can comment out the lines containing the MK library.

If you have [CLion][CLion] installed, this is a simple story. Just open the project and run the code using the provided [CMakeLists.txt file](./CMakeLists.txt). On the right side of the CLion window you can see the available executables.

Otherwise, you can compile the code using the command line.
1. Compile the CMakeFiles:
   ```bash
   cd NLA-challenges # repository folder
   cmake . # where the CMakeLists.txt file is located
   ```
   After running the above command, you will see the Makefile in the repository folder.
   This Makefile contains the necessary commands to compile the code.
   Since the Makefile is generated automatically, you don't need to edit it.
   If you want to edit the Makefile, you can do so by modifying the file [CMakeLists.txt](./CMakeLists.txt).
2. Compile all possible executables with the following command:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . all
   ```
3. And finally, run one of the compiled codes:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   ./challenge_1-main
   # and
   ./challenge_2-main
   ```
4. Clean the compiled files:
   ```bash
   # assuming you are in the repository folder where the CMakeLists.txt file is located
   make -f ./Makefile -C . clean
   ```


[NLA]: https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=837635&__pj0=0&__pj1=c14afe0b1a27f6df8728d3432f9a6132
[POLIMI]: https://www.polimi.it/
[Eigen]: https://eigen.tuxfamily.org/index.php?title=Main_Page
[SVD]: https://en.wikipedia.org/wiki/Singular_value_decomposition
[LIS]: https://www.ssisc.org/lis/
[MK]: https://github.com/pcafrica/mk
[MOX]: https://mox.polimi.it/
[CLion]: https://www.jetbrains.com/clion/
[Eigen-Repo]: https://gitlab.com/libeigen/eigen