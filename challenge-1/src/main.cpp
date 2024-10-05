// from https://github.com/nothings/stb/tree/master
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION

#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

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
    noise_save.join();
    noise_clion_save.join();

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
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }
    for (int i = 0; i < size_vector_einstein_img; ++i) {
        v(i) = static_cast<double>(image_data[i]);
    }
    // Load noise image and create noise image as w vector
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


    /**********
     * Task 6 *
     **********/
    // Create smoothing filter H_av2
    MatrixXd H_sh2 = create_filter(static_cast<matrix_utils::Filter>(sharpening_sh2));
    // Create convolution matrix
    SparseMatrix<double> A2 = create_convolution_matrix(H_sh2, dark_einstein_img);
    // According to the symmetric definition of a matrix: A = A^{T} iff A is symmetric
    const bool A2_is_symmetric = A2.isApprox(A2.transpose());
    printf("\nTask 6. Write the convolution operation corresponding to the sharpening kernel H_{sh2} "
           "as a matrix vector multiplication by a matrix A_{2} having size mn * mn. "
           "Report the number of non-zero entries in A_{2}. Is A_{2} symmetric?\nAnswer: %ld, is A_{2} symmetric? %s\n",
           A2.nonZeros(), A2_is_symmetric ? "true" : "false");


    /**********
     * Task 7 *
     **********/
    // Convolution using the matrix multiplication technique
    VectorXd sharpening_convolution_matrix = A2*v;
    static MatrixXd sharpening_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            const double val = sharpening_convolution_matrix[row_offset + j];
            sharpening_matrix(i, j) = val > 255.0 ? 255.0 : (val < 0 ? 0 : val);
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpening_result(height, width);
    sharpening_result = sharpening_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* sharpening_filename = "sharpening.png";
    const char* clion_sharpening_filename = "../challenge-1/resources/sharpening.png";
    std::thread sharpening_save(
        image_manipulation::save_image_to_file,
        sharpening_filename, width, height, 1, sharpening_result.data(), width
    );
    std::thread sharpening_clione_save(
        image_manipulation::save_image_to_file,
        clion_sharpening_filename, width, height, 1, sharpening_result.data(), width
    );
    sharpening_save.join();
    sharpening_clione_save.join();

    printf(
        "\nTask 7. Apply the previous sharpening filter to the original image by performing the matrix "
        "vector multiplication A_{2}v. Export the resulting image.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(sharpening_filename).c_str(),
        filesystem::absolute(clion_sharpening_filename).c_str()
    );


    /**********
     * Task 8 *
     **********/
    saveMarket(A2, "../challenge-1/resources/A2.mtx");
    // Export vector in .mtx format
    save_market_vector("../challenge-1/resources/w.mtx", w);
    printf(
        "\nExport the Eigen matrix $A_{2}$ and vector $w$ in the .mtx format."
        " Using a suitable iterative solver and preconditioner technique available in the LIS library compute "
        "the approximate solution to the linear system A_{2}x = w prescribing a tolerance of 10^{-9}. "
        "Report here the iteration count and the final residual.\nAnswer: "
    );
    return 0;
}
