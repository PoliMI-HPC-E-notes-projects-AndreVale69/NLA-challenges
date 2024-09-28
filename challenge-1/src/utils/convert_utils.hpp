#ifndef CONVERT_IMPLEMENTATION
#define CONVERT_IMPLEMENTATION

#include <Eigen/Dense>

namespace convert_utils {
    /**
     * Use the tree matrices to compose the RGB representation and convert to a single B/W matrix.
     * @param red Matrix of red colors.
     * @param green Matrix of green colors.
     * @param blue Matrix of blue colors.
     * @return B/W matrix.
     */
    Eigen::MatrixXd convertToGrayscale(const Eigen::MatrixXd& red, const Eigen::MatrixXd& green, const Eigen::MatrixXd& blue);
}

#endif
