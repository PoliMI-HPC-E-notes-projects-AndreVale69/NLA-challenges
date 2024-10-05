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

#include "external_libs/stb_image_write.h"
#include "utils/image_manipulation.hpp"
#include "utils/matrix_utils.hpp"

using namespace std;
using namespace Eigen;
using namespace matrix_utils;


// auto createConvolutionMatrixv3(const int m, const int n) {
//     const int size = m * n;
//     SparseMatrix<double> A1(size, size);
//     pmr::vector<Triplet<double>> tripletList;
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
//                         tripletList.emplace_back(row, col, kernel(ki + 1, kj + 1));
//                     }
//                 }
//             }
//         }
//     }
//
//     A1.setFromTriplets(tripletList.begin(), tripletList.end());
//     return A1;
// }


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
//     convolution_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
//
//     return convolution_matrix;
// }

// TODO: tmp workaround to include the enum
enum Filter: short {
    smoothing_av1,
    smoothing_av2,
    sharpening_sh1,
    sharpening_sh2,
    sobel_vertical_ed1,
    sobel_horizontal_ed2,
    laplacian_edge_lap
};

SparseMatrix<double> create_convolution_matrix_v2(const Matrix<double, 3, 3> &filter, const MatrixXd &matrix) {
    bool almost_one_col_valid = false;
    int rows_filled = 0, rows_offset = 0, offset_filter = 0,
        row_lower_bound_neighbour = 0, row_upper_bound_neighbour = 0,
        col_left_bound_neighbour = 0, col_right_bound_neighbour = 0;
    // const allocation of rows and cols of the original matrix, and matrix size;
    // this avoids multiple memory accesses
    const int matrix_rows = static_cast<int>(matrix.rows());
    const int matrix_cols = static_cast<int>(matrix.cols());
    const long matrix_size = matrix.size();
    // use small array to take the values of the filter;
    // the size is so small that accessing an element n is very fast (O(n))
    // and negligible (in terms of performance)
    const double * filter_array = filter.data();
    // create the convolution (sparse) matrix and the triplet
    SparseMatrix<double> convolution_matrix(matrix_size, matrix_size);
    std::vector<Triplet<double>> triplet_list;
    triplet_list.reserve(filter.size() * matrix_size);

    // for each row of the matrix
    for (int row = 0; row < matrix_rows; ++row) {
        row_upper_bound_neighbour = row - 1;
        row_lower_bound_neighbour = row + 1;
        // for each column of the matrix
        for (int col = 0; col < matrix_cols; ++col, ++rows_filled) {
            offset_filter = 0;
            rows_offset = 0;
            /**
             * check the neighbours:
             * x x x
             * x o x
             * x x x
             * where o is the centre of the filter and the x's are its neighbours;
             * set the new column boundaries before the check
             */
            col_left_bound_neighbour = col-1;
            col_right_bound_neighbour = col+1;
            for (int i_row = row_upper_bound_neighbour; i_row <= row_lower_bound_neighbour; ++i_row) {
                // reset the flag
                almost_one_col_valid = false;
                for (int j_col = col_left_bound_neighbour; j_col <= col_right_bound_neighbour; ++j_col, ++offset_filter) {
                    // check that the index neighbour is valid;
                    // this is essential when the filter is applied to the edge of the matrix
                    if (isIndexOutOfBounds(matrix, i_row, j_col)) {
                        continue;
                    }
                    almost_one_col_valid = true;
                    // optimization to avoid garbage (zero) values; add iff > 0;
                    // use some available memory to store zeros;
                    // this should increase speed, but is it really necessary?
                    if (const auto filter_value = filter_array[offset_filter]; filter_value > 0.0) {
                        triplet_list.emplace_back(rows_filled, j_col+rows_offset, filter_value);
                    }
                }
                if (almost_one_col_valid) {
                    rows_offset += matrix_cols;
                }
            }
        }
    }
    // create the new sparse matrix using the triplet;
    // hey garbage collector, the triplet will be all yours soon!
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
        "Verify that each vector has m n components. Report here the Euclidean norm of v.\nAnswer: %f\n", v.norm()
    );


    /**********
     * Task 4 *
     **********/
    // Create smoothing filter H_av2
    MatrixXd H_av2 = create_filter(static_cast<matrix_utils::Filter>(smoothing_av2));

    // Create convolution matrix
    MatrixXd debug(5, 4);
    int counter = 1;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 4; ++j) {
            counter++;
            debug(i, j) = static_cast<double>(counter);
        }
    }
    // SparseMatrix<double> A2 = create_convolution_matrix_v2(H_av2, dark_einstein_img);
    SparseMatrix<double> A2 = create_convolution_matrix_v2(H_av2, debug);

    cout << "\nTask 4.Write the convolution operation corresponding to the smoothing kernel H_{av2} "
           "as a matrix vector multiplication between a matrix A_{1} having size mn times mn and the image vector."
           "Report the number of non-zero entries in A_{1}.\nAnswer: \n" << A2.toDense();
    // printf("\nTask 4.Write the convolution operation corresponding to the smoothing kernel H_{av2} "
    //        "as a matrix vector multiplication between a matrix A_{1} having size mn times mn and the image vector."
    //        "Report the number of non-zero entries in A_{1}.\nAnswer: %ld\n", A2.toDense());


    /**********
     * Task 5 *
     **********/

    // task 5 :
    // VectorXd smoothed_img = A2*w;
    // int size = height * width;
    // // vector<unsigned char> out_smoothed(size);
    //
    // // for(int i = 0; i<size; ++i){
    // //   out_smoothed[i] = static_cast<unsigned char>(smoothed_img[i]);
    // // }
    //
    // Matrix<unsigned char, Dynamic, Dynamic, RowMajor> out_smoothed = smoothed_img.unaryExpr([](const double val) -> unsigned char {
    //   return static_cast<unsigned char>(val);
    // });
    //
    // image_manipulation::save_image_to_file("smoothing.png", width, height, 1, out_smoothed.data(), width);

    // Product<SparseMatrix<double>, VectorXd> res = A2 * w;
    // MatrixXd einstein_smoothing(height, width);
    // vector<Triplet<double>> triplets;
    // triplets.reserve(A2.nonZeros());
    //
    // row_offset = 0;
    // for (int i = 0; i < height; ++i) {
    //     for (int j = 0; j < width; ++j) {
    //         triplets.emplace_back(i, j, res[row_offset+j]);
    //         // einstein_smoothing(i, j) = res.operator()(row_offset+j);// res.coeff(row_offset+j);
    //     }
    //     row_offset += width;
    // }
    // SparseMatrix<double> prova;
    // prova.setFromTriplets(triplets.begin(), triplets.end());
    //
    // Matrix<unsigned char, Dynamic, Dynamic, RowMajor> convolution_try = prova.unaryExpr([](const double val) -> unsigned char {
    //   return static_cast<unsigned char>(val);
    // });
    //
    // // Save the image using stbi_write_jpg
    // const char* filename = "convolution_try.png";
    // image_manipulation::save_image_to_file(filename, width, height, 1, convolution_try.data(), width);
    //
    // // Print Task 5
    // printf(
    //     "\nAnswer: see the figure %s\n",
    //     filesystem::absolute(filename).c_str()
    // );

    return 0;
}
