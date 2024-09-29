#include "image_manipulation.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../external_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external_libs/stb_image_write.h"

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

    void save_image_to_file(
        const char *filename, const int x, const int y, const int comp, const void *data, const int stride_bytes
    ) {
        if (stbi_write_png(filename, x, y, comp, data, stride_bytes) == 0) {
            throw std::runtime_error("Failed to write noise.png image.");
        }
    }
}
