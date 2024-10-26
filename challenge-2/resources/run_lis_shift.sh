#!/bin/bash
source /u/sw/etc/profile
module load gcc-glibc
module load lis
cd ../challenge-2/resources || cd ../resources || exit
./eigen1 ATA.mtx eigvec_shift.mtx hist_shift.txt -e pi -etol 1.0e-8 -shift 4.244525e+07 -emaxiter 1000 > result_shift.txt
