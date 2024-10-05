// from https://github.com/nothings/stb/tree/master
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "utils/image_manipulation.hpp"
#include "utils/matrix_utils.hpp"

using namespace std;
using namespace Eigen;
using namespace matrix_utils;

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

SparseMatrix<double> create_convolution_matrix(const Matrix<double, 3, 3> &filter, const MatrixXd &matrix) {
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
                        if (row >= 2) {
                            triplet_list.emplace_back(rows_filled, j_col+rows_offset+(row-1)*matrix_cols, filter_value);
                            continue;
                        }
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
    unsigned char* image_data, * noise_image_data;
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_img(height, width);
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
    const char* clion_noise_filename = "../challenge-1/resources/noise.png";
    // thread feature to speedup I/O operations
    std::thread noise_save(
        image_manipulation::save_image_to_file,
        noise_filename, width, height, 1, einstein_noise.data(), width
    );
    std::thread noise_clion_save(
        image_manipulation::save_image_to_file,
        clion_noise_filename, width, height, 1, einstein_noise.data(), width
    );

    printf(
        "\nTask 2. Introduce a noise signal into the loaded image "
        "by adding random fluctuations of color ranging between [-50, 50] to each pixel. "
        "Export the resulting image in .png and upload it.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(noise_filename).c_str(),
        filesystem::absolute(clion_noise_filename).c_str()
    );


    /**********
     * Task 3 *
     **********/
    // Create original image as v vector
    const long size_vector_einstein_img = dark_einstein_img.cols() * dark_einstein_img.rows();
    VectorXd v(size_vector_einstein_img);
    for (int i = 0; i < size_vector_einstein_img; ++i) {
        v(i) = static_cast<double>(image_data[i]);
    }
    // Load noise image and create noise image as w vector
    noise_save.join();
    noise_clion_save.join();
    try {
        noise_image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/noise.png", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        noise_image_data = image_manipulation::load_image_from_file(
            "../resources/noise.png", width, height, channels
        );
    }
    const long size_vector_noise_img = dark_noise_img.cols() * dark_noise_img.rows();
    VectorXd w(size_vector_noise_img);
    for (int i = 0; i < size_vector_noise_img; ++i) {
        w(i) = static_cast<double>(noise_image_data[i]);
    }

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
    SparseMatrix<double> A1 = create_convolution_matrix(H_av2, dark_einstein_img);
    printf("\nTask 4.Write the convolution operation corresponding to the smoothing kernel H_{av2} "
           "as a matrix vector multiplication between a matrix A_{1} having size mn times mn and the image vector."
           "Report the number of non-zero entries in A_{1}.\nAnswer: %ld\n", A1.nonZeros());


    /**********
     * Task 5 *
     **********/
    // Convolution using the matrix multiplication technique
    VectorXd convolution_multiplication = A1*w;
    static MatrixXd smoothing_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            smoothing_matrix(i, j) = convolution_multiplication[row_offset + j];
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> result(height, width);
    result = smoothing_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* smoothing_filename = "smoothing.png";
    const char* clion_smoothing_filename = "../challenge-1/resources/smoothing.png";
    std::thread smoothing_save(
        image_manipulation::save_image_to_file,
        smoothing_filename, width, height, 1, result.data(), width
    );
    std::thread smoothing_clion_save(
        image_manipulation::save_image_to_file,
        clion_smoothing_filename, width, height, 1, result.data(), width
    );
    smoothing_save.join();
    smoothing_clion_save.join();

    printf(
    "\nTask 5. Apply the previous smoothing filter to the noisy image by performing "
    "the matrix vector multiplication A_{1}w Export the resulting image.\nAnswer: see the figure %s\nAnd: %s\n",
    filesystem::absolute(smoothing_filename).c_str(),
    filesystem::absolute(clion_smoothing_filename).c_str()
    );

    return 0;
}
