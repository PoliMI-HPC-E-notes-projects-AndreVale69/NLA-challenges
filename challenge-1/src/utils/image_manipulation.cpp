#include "image_manipulation.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../external_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external_libs/stb_image_write.h"

#include <iostream>
#include <sstream>


namespace image_manipulation {
    unsigned char *load_image_from_file(int &width, int &height, int &channels) {
        const char* input_img_path = "../challenge-1/resources/Albert_Einstein_Head.jpg";
        unsigned char* image_data = stbi_load(input_img_path, &width, &height, &channels, 1);
        if (!image_data) {
            image_data = stbi_load(
                "../resources/Albert_Einstein_Head.jpg", &width, &height, &channels, 1
            );
            if (!image_data) {
                throw std::runtime_error("Could not load image Albert_Einstein_Head.jpg");
            }
        }
        std::cout << "Image loaded: " << width << " x " << height << " with " << channels << " channels\n";
        stbi_image_free(image_data);
        return image_data;
    }

    void save_image_to_file(
        const char *filename, const int x, const int y, const int comp, const void *data, const int stride_bytes
    ) {
        if (stbi_write_png(filename, x, y, comp, data, stride_bytes) == 0) {
            const std::stringstream error{"Failed to write "};
            error.str().append(filename);
            throw std::runtime_error(error.str());
        }
    }
}
