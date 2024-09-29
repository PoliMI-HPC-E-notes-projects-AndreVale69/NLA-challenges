// from https://github.com/nothings/stb/tree/master
// #define STB_IMAGE_IMPLEMENTATION
// #include "external_libs/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "external_libs/stb_image_write.h"

#include <iostream>
#include <random>
#include <Eigen/Sparse>

#include "utils/image_manipulation.hpp"


int main() {
    /**********
     * Task 1 *
     **********/
    // Load image from file
    int width = 0, height = 0, channels = 0;
    unsigned char* image_data = image_manipulation::load_image_from_file(width, height, channels);

    // Get matrix from image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> einstein_img(height, width);
    image_manipulation::convert_bw_image_to_matrix(width, height, channels, image_data, einstein_img);

    // Print Task 1
    std::cout << "\nTask 1. Load the image as an Eigen matrix with size m*n. "
                 "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere "
                 "between 0 (black) and 255 (white). "
                 "Report the size of the matrix.\n"
                 "Answer: " << einstein_img.rows() << " * " << einstein_img.cols() << " (rows * cols)\n";

    /**********
     * Task 2 *
     **********/
    // Create a random device and a Mersenne Twister random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> d(-50, 50);

    // Create result matrix
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> einstein_noise(height, width);
    Eigen::MatrixXd dark(height, width), light(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const int index = (i * width + j) * channels;  // 1 channel (Greyscale)
            double new_value = static_cast<double>(image_data[index]) + d(gen);
            new_value = new_value > 255.0 ? 255.0 : (new_value < 0 ? 0 : new_value);
            dark(i, j) = new_value;
            light(i, j) = new_value;
            // Less optimized implementation
            // dark(i, j) = std::max(std::min(static_cast<double>(image_data[index]) + d(gen), 255.), 0.);
            // light(i, j) = std::max(std::min(static_cast<double>(image_data[index]) + d(gen), 255.), 0.);
        }
    }

    // Use Eigen's unaryExpr to map the grayscale values
    einstein_noise = dark.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Save the image using stbi_write_jpg
    const char* noise_filename = "noise.png";
    image_manipulation::save_image_to_file("noise.png", width, height, 1, einstein_noise.data(), width);

    // Print Task 2
    std::cout << "\nTask 2. Introduce a noise signal into the loaded image "
                 "by adding random fluctuations of color ranging between [-50, 50] to each pixel. "
                 "Export the resulting image in .png and upload it.\n"
                 "Answer: " << "see the figure call " << noise_filename << '\n';

    return 0;
}
