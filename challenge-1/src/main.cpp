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

/**
 * Take an Eigen MatrixXd and return a sparse matrix with zero padding at the boundaries.
 * The sparse matrix structure is used to avoid wasting space.
 * @param matrix matrix to pad.
 * @return padded original matrix.
 */
SparseMatrix<float> zero_padding(const MatrixXd &matrix) {
    const long rows = matrix.rows();
    const long cols = matrix.cols();
    const long zero_padding_rows = rows + 2;
    const long zero_padding_cols = cols + 2;
    SparseMatrix<float> zero_padding_matrix(zero_padding_rows, zero_padding_cols);
    std::vector<Triplet<float>> triplet_list;
    // pre allocation optimization
    triplet_list.reserve(matrix.size());
    for (int r_matrix = 0, r_zero_pad_matrix = 1; r_matrix < rows; ++r_matrix, ++r_zero_pad_matrix) {
        Matrix<double, -1, -1> row_matrix = matrix.row(r_matrix);
        for (int c_matrix = 0, c_zero_pad_matrix = 1; c_matrix < cols; ++c_matrix, ++c_zero_pad_matrix) {
            triplet_list.emplace_back(r_zero_pad_matrix, c_zero_pad_matrix, row_matrix(c_matrix));
        }
    }
    zero_padding_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    // std::cout << "\nRow0:\n" << zero_padding_matrix.row(0) << '\n';
    // std::cout << "\nRown:\n" << zero_padding_matrix.row(rows+1) << '\n';
    // std::cout << "\nCol0:\n" << zero_padding_matrix.col(0) << '\n';
    // std::cout << "\nColn:\n" << zero_padding_matrix.col(cols+1) << '\n';
    return zero_padding_matrix;
}

// SparseMatrix<float> create_convolution_matrix(const Matrix<float, 3, 3> &filter, const MatrixXd &matrix) {
//     const SparseMatrix<float> padded_matrix = zero_padding(matrix);
//     const long filter_rows = filter.rows();
//     const long filter_cols = filter.cols();
//     const long matrix_rows = matrix.rows();
//     const long matrix_cols = matrix.cols();
//     const long padded_matrix_cols = padded_matrix.cols();
//     // const long convolution_matrix_rows = matrix_cols * matrix_rows;
//     const long convolution_matrix_rows = padded_matrix.size();
//     const long number_of_zeros = matrix_cols - filter_cols;
//     SparseMatrix<float> convolution_matrix(convolution_matrix_rows, convolution_matrix_rows);
//     std::vector<Triplet<float>> triplet_list;
//     // pre allocation optimization
//     // size of the filter (because each row of the convolution matrix contains each entry of the filter)
//     // times the rows of the convolution matrix to build
//     // (then the convolution matrix will have a size of rows*cols * rows*cols)
//     triplet_list.reserve(filter.size() * convolution_matrix_rows);
//
//
//     int num_entries = 0;
//     // the number of rows in the convolution matrix filled with the filter value
//     // in the end, it should be equal to the rows of the convolution matrix
//     int filled_rows = 0;
//     // in each row of the convolution matrix,
//     // the column where the filter values start to be placed is summed up to the offset.
//     // trivially, it is the number of columns in the filled matrix times the number of rows already filled
//     long offset = 0;
//     long row_offset = 0;
//     // row_done is the number of rows already filled
//     // (in the convolution matrix) with respect to the original matrix.
//     // the value is increased each time n rows are filled with filter values
//     // (where n is the number of columns in the original matrix).
//     // the for iteration continues until each row of the matrix is no longer convolutional.
//     for (int row_done = 0; row_done < matrix_rows; ++row_done) {
//         // for each element of the original matrix row,
//         // create i rows in the convolution matrix, with each row filled with filter values
//         for (int i = 0; i < matrix_cols; ++i, ++filled_rows) {
//             row_offset = 0;
//             for (int blocks = 0; blocks < filter_rows; ++blocks) {
//                 for (num_entries = 0; num_entries < filter_cols; ++num_entries) {
//                     triplet_list.emplace_back(filled_rows, i+offset+row_offset+num_entries, filter.coeff(blocks, num_entries));
//                 }
//                 row_offset += num_entries + number_of_zeros;
//             }
//         }
//         offset += padded_matrix_cols;
//     }
//     // assert(filled_rows == convolution_matrix_rows);
//     // TODO: fix index!
//     convolution_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
//
//     return convolution_matrix;
// }

bool isIndexOutOfBounds(const MatrixXd& matrix, const int row, const int col) {
    return row < 0 || row >= matrix.rows() || col < 0 || col >= matrix.cols();
}

SparseMatrix<double> create_convolution_matrix_v2(const Matrix<float, 3, 3> &filter, const MatrixXd &matrix) {
    const int matrix_rows = static_cast<int>(matrix.rows());
    const int matrix_cols = static_cast<int>(matrix.cols());
    SparseMatrix<double> convolution_matrix(matrix.size(), matrix.size());
    std::vector<Triplet<double>> triplet_list;
    triplet_list.reserve(filter.size() * matrix_rows * matrix_cols);

    int rows_filled = 0;
    int rows_offset = 0;
    int offset_filter = 0;
    int j_col = 0;
    bool almost_one_col_valid = false;
    const auto filter_array = filter.array();

    for (int row = 0; row < matrix_rows; ++row) {
        // lower_row_offset = row-1;
        // upper_row_offset = row+1;
        for (int col = 0; col < matrix_cols; ++col, ++rows_filled) {
            offset_filter = 0;
            rows_offset = 0;
            for (int i_row = row-1; i_row < row+1+1; ++i_row) {
                almost_one_col_valid = false;
                for (j_col = col-1; j_col < col+1+1; ++j_col, ++offset_filter) {
                    if (isIndexOutOfBounds(matrix, i_row, j_col)) {
                        continue;
                    }
                    almost_one_col_valid = true;
                    // optimization to avoid garbage values; add iff > 0
                    if (const auto filter_value = filter_array.operator()(offset_filter); filter_value > 0.0) {
                        triplet_list.emplace_back(rows_filled, j_col+rows_offset, filter_value);
                    }
                }
                if (almost_one_col_valid) {
                    rows_offset += matrix_cols;
                }
            }
        }
    }
    convolution_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return convolution_matrix;
}

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
    Matrix<float, 3, 3> H_av2 = static_cast<float>(1) / static_cast<float>(9) * Matrix<float, 3, 3>::Ones(3,3);

    // SparseMatrix<double, 0, long int> padded_dark_einstein_img(dark_einstein_img.rows()+2, dark_einstein_img.cols()+2);
    // padded_dark_einstein_img.block<static_cast<int>(dark_einstein_img.rows()), static_cast<int>(dark_einstein_img.cols())>(1,1) = dark_einstein_img.sparseView();

    SparseMatrix<double> convolution_matrix = create_convolution_matrix_v2(H_av2, dark_einstein_img);
    // SparseMatrix<float> convolution_matrix = create_convolution_matrix(H_av2, dark_einstein_img);

    printf("\nNNZ: %ld\n", convolution_matrix.nonZeros());

    auto res = convolution_matrix * w;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> prova = res.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Save the image using stbi_write_jpg
    const char* filename = "prova.png";
    image_manipulation::save_image_to_file(filename, width, height, 1, prova.data(), width);

    // Print Task 2
    printf(
        "\nAnswer: see the figure %s\n",
        filesystem::absolute(filename).c_str()
    );

    // SparseMatrix<long, ColMajor, long> A1(size_vector_einstein_img, size_vector_einstein_img);
    // auto A1 = createConvolutionMatrix(height, width);
    // cout << "\nResult: " << A1.rows() << " * " << A1.cols() << " nnz: " << A1.nonZeros() << "\n";

    // Create result matrix
    // auto resultVector = convolution_matrix * w;
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
