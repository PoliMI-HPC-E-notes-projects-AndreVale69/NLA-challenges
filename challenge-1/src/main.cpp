// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "external_libs/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "external_libs/stb_image_write.h"

#include <iostream>
#include <Eigen/Sparse>

#include "utils/image_manipulation.hpp"


int main() {
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

    /*
    // Save the image using stbi_write_jpg
    const std::string output_image_path1 = "dark_image.jpg";
    stbi_write_jpg(output_image_path1.c_str(), width, height, 1, einstein_img.data(), 100);
    if (stbi_write_jpg(
        output_image_path1.c_str(), width, height, 1, einstein_img.data(), 100
        ) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;
        return 1;
    }
    */

    return 0;
}
