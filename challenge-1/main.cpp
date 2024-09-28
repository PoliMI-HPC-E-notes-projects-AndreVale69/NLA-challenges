#include <iostream>
#include <filesystem>
#include <Eigen/Sparse>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using std::filesystem::current_path;

// Function to convert RGB to grayscale
Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd& red, const Eigen::MatrixXd& green,
                                   const Eigen::MatrixXd& blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

int main() {
    // Load the image using stb_image
    const char* INPUT_IMG_PATH = "../challenge-1/Albert_Einstein_Head.jpg";
    int width, height, channels;
    unsigned char* image_data = stbi_load(INPUT_IMG_PATH, &width, &height,
                                          &channels, 3);  // Force load as RGB
    if (!image_data) {
        std::cerr << "Error: Could not load image Albert_Einstein_Head.jpg" << '\n';
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with "
              << channels << " channels." << '\n';

    // Prepare Eigen matrices for each RGB channel
    Eigen::MatrixXd red(height, width), green(height, width), blue(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int index = (i * width + j) * 3;  // 3 channels (RGB)
            red(i, j) = static_cast<double>(image_data[index]) / 255.0;
            green(i, j) = static_cast<double>(image_data[index + 1]) / 255.0;
            blue(i, j) = static_cast<double>(image_data[index + 2]) / 255.0;
        }
    }
    // Free memory
    stbi_image_free(image_data);

    // Create a grayscale matrix
    Eigen::MatrixXd gray = convertToGrayscale(red, green, blue);

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grayscale_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    grayscale_image = gray.unaryExpr([](double val) -> unsigned char {
      return static_cast<unsigned char>(val * 255.0);
    });

    std::cout << "1. Load the image as an Eigen matrix with size m*n. "
                 "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere "
                 "between 0 (black) and 255 (white)."
                 "Report the size of the matrix." << '\n' <<
                 "Answer: " << grayscale_image.rows() << " * " << grayscale_image.cols() << " (rows * cols)" << '\n';

    // Save the grayscale image using stb_image_write
    /*
    const std::string output_image_path = "output_grayscale.png";
    if (stbi_write_png(output_image_path.c_str(), width, height, 1,
                       grayscale_image.data(), width) == 0) {
        std::cerr << "Error: Could not save grayscale image" << std::endl;

        return 1;
                       }

    std::cout << "Grayscale image saved to " << output_image_path << std::endl;
    */

    return 0;
}
