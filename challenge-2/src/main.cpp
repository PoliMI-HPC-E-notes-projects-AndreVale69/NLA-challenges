#ifdef __CYGWIN__
    std::cout << "The script has been tested on one of the Debian distributions, we don't guarantee that it will work."
#endif
#ifdef __APPLE__
    std::cout << "The script has been tested on one of the Debian distributions, we don't guarantee that it will work."
#endif


#include <iostream>
#include "utils/matrix_utils.hpp"

#include "utils/image_manipulation.hpp"
#include "utils/matrix_utils.hpp"

using namespace std;
using namespace Eigen;
using namespace matrix_utils;



int main() {
    /**********
     * Task 1 *
     **********/
    // Load image from file
    int width = 0, height = 0, channels = 0, row_offset = 0;
    unsigned char* image_data;
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-2/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_img(height, width);
    static MatrixXd einstein_matrix(height, width),
                    einstein_matrix_transpose(height, width),
                    einstein_matrix_transpose_times_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            einstein_matrix(i, j) = static_cast<double>(image_data[row_offset + j]);
        }
    }
    einstein_matrix_transpose = einstein_matrix.transpose();
    einstein_matrix_transpose_times_matrix = einstein_matrix_transpose * einstein_matrix;

    printf(
    "\nTask 1. Load the image as an Eigen matrix with size m * n."
        "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0"
        "(black) and 255 (white)."
        "Compute the matrix product A^{T}A and report the Euclidean norm of A^{T}A.\n"
        "Answer: %f\n", einstein_matrix_transpose_times_matrix.norm()
    );


    /**********
     * Task 2 *
     **********/
    // SelfAdjointEigenSolver
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    const SelfAdjointEigenSolver<MatrixXd> eigensolver(einstein_matrix_transpose_times_matrix);
    // Check all is ok
    if (eigensolver.info() != Success) {
        abort();
    }
    // Take eigenvalues "the eigenvalues are sorted in increasing order"
    const VectorXd& eigenvalues = eigensolver.eigenvalues();
    const long eigenvalues_size = eigenvalues.size();
    printf(
        "\nTask 2. Solve the eigenvalue problem A^{T}Ax = lambda x using the proper solver"
        "provided by the Eigen library. Report the two largest computed singular values of A.\n"
        "Answer: %f (largest), %f (second largest)\n",
        eigenvalues.coeff(eigenvalues_size-1), eigenvalues.coeff(eigenvalues_size-2)
    );

    return 0;
}
