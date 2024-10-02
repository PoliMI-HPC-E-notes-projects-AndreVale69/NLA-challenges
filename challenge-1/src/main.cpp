// from https://github.com/nothings/stb/tree/master
// #define STB_IMAGE_IMPLEMENTATION
#include "external_libs/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "external_libs/stb_image_write.h"

#include <filesystem>
#include <iostream>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "utils/image_manipulation.hpp"

using namespace std;
using namespace Eigen;

int main() {
    /**********
     * Task 1 *
     **********/
    // Load image from file
    int width = 0, height = 0, channels = 0, row_offset = 0;
    unsigned char* image_data = image_manipulation::load_image_from_file(width, height, channels);

    // Get matrix from image
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_img(height, width);

    // Prepare Eigen matrix
    static MatrixXd dark_einstein_img(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            dark_einstein_img(i, j) = static_cast<double>(image_data[row_offset + j]);
        }
    }
    // Free memory
    stbi_image_free(image_data);
    einstein_img = dark_einstein_img.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Print Task 1
    printf(
        "\nTask 1. Load the image as an Eigen matrix with size m*n. "
        "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere "
        "between 0 (black) and 255 (white). Report the size of the matrix.\n"
        "Answer: %ld * %ld (rows * cols)\n", einstein_img.rows(), einstein_img.cols()
    );


    /**********
     * Task 2 *
     **********/
    // Create a random device and a Mersenne Twister random number generator
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution d(-50, 50);

    // Create result matrix
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_noise(height, width);
    MatrixXd dark_noise_img(height, width);
    int new_value = 0;

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            new_value = image_data[row_offset + j] + d(gen);
            dark_noise_img(i, j) = new_value > 255.0 ? 255.0 : (new_value < 0 ? 0 : new_value);
        }
    }

    // Use Eigen's unaryExpr to map the grayscale values
    einstein_noise = dark_noise_img.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Save the image using stbi_write_jpg
    const char* noise_filename = "noise.png";
    image_manipulation::save_image_to_file(noise_filename, width, height, 1, einstein_noise.data(), width);

    // Print Task 2
    printf(
        "\nTask 2. Introduce a noise signal into the loaded image "
        "by adding random fluctuations of color ranging between [-50, 50] to each pixel. "
        "Export the resulting image in .png and upload it.\nAnswer: see the figure %s\n",
        filesystem::absolute(noise_filename).c_str()
    );


    /**********
     * Task 3 *
     **********/
    // Create original image as v vector
    const long size_vector_einstein_img = dark_einstein_img.cols() * dark_einstein_img.rows();
    // manually:
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         v(i * width + j) = dark_einstein_img(i, j);
    //     }
    // }
    VectorXd v(Map<VectorXd>(dark_einstein_img.data(), size_vector_einstein_img));
    // Create noise image as w vector
    const long size_vector_noise_img = dark_noise_img.cols() * dark_noise_img.rows();
    // manually:
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         w(i * width + j) = dark_noise_img(i, j);
    //     }
    // }
    VectorXd w(Map<VectorXd>(dark_noise_img.data(), size_vector_noise_img));

    // Verify that each vector has m*n components
    if (v.size() != size_vector_einstein_img || w.size() != size_vector_noise_img) {
        stringstream error;
        error << "Convertion from matrix to vector failed.\n" <<
            "Vector v size: " << v.size() << ", expected size: " << size_vector_einstein_img << '\n' <<
            "Vector w size: " << w.size() << ", expected size: " << size_vector_noise_img << '\n';
        throw runtime_error(error.str());
    }

    printf(
        "\nTask 3. Reshape the original and noisy images as vectors v and w, respectively. "
        "Verify that each vector has m n components. Report here the Euclidean norm of v.\nAnswer: %f", v.norm()
    );


    /**********
     * Task 4 *
     **********/
    // Create smoothing filter H_av2
    MatrixXd H_av2 = (static_cast<float>(1) / static_cast<float>(9)) * MatrixXd::Ones(3,3);
    // SparseMatrix<long, ColMajor, long> A1(size_vector_einstein_img, size_vector_einstein_img);
    // auto A1 = createConvolutionMatrix(height, width);
    // cout << "\nResult: " << A1.rows() << " * " << A1.cols() << " nnz: " << A1.nonZeros() << "\n";

    // Create result matrix
    // VectorXd resultVector = A1 * w;
    // Matrix<unsigned char, Dynamic, Dynamic, RowMajor> A1w;
    // MatrixXd A1w_dark(height, width), A1w_light(height, width);
    //
    // // Fill the matrices with image data
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         const int index = (i * width + j) * channels;  // 1 channel (Greyscale)
    //         A1w_dark(i, j) = static_cast<double>(image_data_mod[index]);
    //         A1w_light(i, j) = static_cast<double>(image_data_mod[index]);
    //     }
    // }
    //
    // // Use Eigen's unaryExpr to map the grayscale values
    // A1w = A1w_dark.unaryExpr([](const double val) -> unsigned char {
    //   return static_cast<unsigned char>(val);
    // });
    //
    // // Save the image using stbi_write_jpg
    // const char* a1w_png = "a1w.png";
    // save_image_to_file(a1w_png, width, height, 1, A1w.data(), width);

    // cout << "Result: " << H_av2.colPivHouseholderQr().solve(einstein_img).nonZeros() << '\n';
    // cout << "Result: " << (Map<VectorXd>(H_av2.data(), size_vector_einstein_img)).colPivHouseholderQr().solve(einstein_img).nonZeros() << '\n';

    return 0;
}

// auto createConvolutionMatrix(const int m, const int n) {
//     const int size = m * n;
//     SparseMatrix<long> A1(size, size);
//     pmr::vector<Triplet<long>> tripletList;
//     tripletList.reserve(size * 9);
//
//     auto kernel = (static_cast<float>(1) / static_cast<float>(9)) * MatrixXd::Ones(3,3);
//
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             int row = i * n + j;
//             for (int ki = -1; ki <= 1; ++ki) {
//                 for (int kj = -1; kj <= 1; ++kj) {
//                     int ni = i + ki;
//                     int nj = j + kj;
//                     if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
//                         int col = ni * n + nj;
//                         tripletList.push_back(Triplet<long>(row, col, kernel(ki + 1, kj + 1)));
//                     }
//                 }
//             }
//         }
//     }
//
//     A1.setFromTriplets(tripletList.begin(), tripletList.end());
//     return A1;
// }
