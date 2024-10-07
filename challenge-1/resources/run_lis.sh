#!/bin/bash
source /u/sw/etc/profile
module load gcc-glibc
module load lis
cd ../challenge-1/resources || cd ../resources || exit
# the iteration method is CGS (index 3) an improvement to the biconjugate gradient method
# the preconditioner is the Crot ILU (Incomplete LU), a variant of the ILU factorization technique;
# we decide to use this preconditioner because we are using sparse matrices and also
# it is known as one of the best (robust) and commonly used as preconditioner method;
# an interesting paper about Crot ILU implementation of the University of Minnesota:
# https://www-users.cse.umn.edu/~saad/PDF/umsi-2002-021.pdf
./test1 A2.mtx w.mtx sol.mtx hist.txt -tol 1.0e-9 -i cgs -p 8 -maxiter 100 > result.txt
