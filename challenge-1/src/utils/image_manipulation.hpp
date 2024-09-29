#ifndef CONVERT_IMPLEMENTATION
#define CONVERT_IMPLEMENTATION
#include <Eigen/Sparse>

namespace image_manipulation {
    /**
     * Load Einstein figure from file.
     * @param width width of the figure.
     * @param height height of the figure.
     * @param channels channels of the figure.
     * @return The 'unsigned char *' which points to the pixel data,
     * or NULL on an allocation failure or if the image is corrupt or invalid.
     */
    unsigned char *load_image_from_file(int &width, int &height, int &channels);

    /**
     * Save an image as a file, the extension should be png.
     * @param filename name of the file to save.
     * @param x width.
     * @param y height.
     * @param comp compression level.
     * @param data a pointer to the data array of the figure matrix.
     * @param stride_bytes stride bytes.
     */
    void save_image_to_file(char const *filename, int x, int y, int comp, const void *data, int stride_bytes);

    /**
     * Convert a black and white figure to a matrix.
     * @param width width of the figure.
     * @param height height of the figure.
     * @param channels channels of the figure.
     * @param image_data the image data use to convert.
     * @param matrix_result the converted matrix.
     */
    void convert_bw_image_to_matrix(
        int width, int height, int channels,
        unsigned char* &image_data,
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_result
    );

    Eigen::Matrix<unsigned char, -1, -1, 1> create_noise_effect(
        int lower_bound, int upper_bound,
        int width, int height, int channels,
        unsigned char * image_data//,
        //Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &matrix_result
    );
}

#endif
