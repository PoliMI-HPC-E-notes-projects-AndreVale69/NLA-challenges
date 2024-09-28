#include "convert_utils.hpp"
#include <Eigen/Dense>

namespace convert_utils {
    Eigen::MatrixXd convertToGrayscale(
    const Eigen::MatrixXd& red, const Eigen::MatrixXd& green, const Eigen::MatrixXd& blue
    ) {
        return 0.299 * red + 0.587 * green + 0.114 * blue;
    }
}
