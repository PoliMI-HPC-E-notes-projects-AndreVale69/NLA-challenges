#include "image_manipulation.hpp"

#include "../external_libs/stb_image.h"

#include <iostream>


namespace image_manipulation {
    unsigned char *load_image_from_file(int &width, int &height, int &channels) {
        // Load the image using stb_image
        const char* INPUT_IMG_PATH = "../challenge-1/resources/Albert_Einstein_Head.jpeg";
        // int width = 0, height = 0, channels = 0;
        unsigned char* image_data = stbi_load(INPUT_IMG_PATH, &width, &height, &channels, 1);
        if (!image_data) {
            image_data = stbi_load(
                "../resources/Albert_Einstein_Head.jpeg", &width, &height, &channels, 1
            );
            if (!image_data) {
                throw std::runtime_error("Could not load image Albert_Einstein_Head.jpeg");
            }
        }
        std::cout << "Image loaded: " << width << " x " << height << " with " << channels << " channels\n";
        return image_data;
    }

    void convert_bw_image_to_matrix(
        const int width, const int height, const int channels,
        unsigned char* &image_data,
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_result
    ) {
        // Prepare Eigen matrices for each RGB channel
        Eigen::MatrixXd dark(height, width), light(height, width);

        // Fill the matrices with image data
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                const int index = (i * width + j) * channels;  // 1 channel (Greyscale)
                dark(i, j) = static_cast<double>(image_data[index]);// / 255.0;
                light(i, j) = static_cast<double>(image_data[index]); // / 255.0;
            }
        }
        // Free memory
        stbi_image_free(image_data);

        // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
        matrix_result = dark.unaryExpr([](const double val) -> unsigned char {
          return static_cast<unsigned char>(val);// * 255.0);
        });
    };
}
