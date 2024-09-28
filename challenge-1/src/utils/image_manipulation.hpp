#ifndef CONVERT_IMPLEMENTATION
#define CONVERT_IMPLEMENTATION
#include <Eigen/Sparse>

namespace image_manipulation {
    /**
     * Load Einstein figure from file.
     * @return The 'unsigned char *' which points to the pixel data,
     * or NULL on an allocation failure or if the image is corrupt or invalid.
     */
    unsigned char *load_image_from_file(int &width, int &height, int &channels);

    void convert_bw_image_to_matrix(
        int width, int height, int channels,
        unsigned char* &image_data,
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_result
    );
}

#endif
