#include "matrix_utils.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace matrix_utils {
    Eigen::SparseMatrix<float> zero_padding(const Eigen::MatrixXd &matrix) {
        const long rows = matrix.rows();
        const long cols = matrix.cols();
        const long zero_padding_rows = rows + 2;
        const long zero_padding_cols = cols + 2;
        Eigen::SparseMatrix<float> zero_padding_matrix(zero_padding_rows, zero_padding_cols);
        std::vector<Eigen::Triplet<float>> triplet_list;
        // pre allocation optimization
        triplet_list.reserve(matrix.size());
        for (int r_matrix = 0, r_zero_pad_matrix = 1; r_matrix < rows; ++r_matrix, ++r_zero_pad_matrix) {
            Eigen::Matrix<double, -1, -1> row_matrix = matrix.row(r_matrix);
            for (int c_matrix = 0, c_zero_pad_matrix = 1; c_matrix < cols; ++c_matrix, ++c_zero_pad_matrix) {
                triplet_list.emplace_back(r_zero_pad_matrix, c_zero_pad_matrix, row_matrix(c_matrix));
            }
        }
        zero_padding_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
        return zero_padding_matrix;
    }

    bool isIndexOutOfBounds(const Eigen::MatrixXd &matrix, const int row, const int col) {
        return row < 0 || row >= matrix.rows() || col < 0 || col >= matrix.cols();
    }

    enum Filter: short {
        smoothing_av1,
        smoothing_av2,
        sharpening_sh1,
        sharpening_sh2,
        sobel_vertical_ed1,
        sobel_horizontal_ed2,
        laplacian_edge_lap
    };

    Eigen::MatrixXd create_filter(const Filter filter_name) {
        Eigen::MatrixXd filter_matrix = Eigen::MatrixXd::Zero(3, 3);
        switch (filter_name) {
            case smoothing_av1:
                filter_matrix(0, 1) = 1.0;
                filter_matrix(1, 0) = 1.0;
                filter_matrix(1, 1) = 4.0;
                filter_matrix(1, 2) = 1.0;
                filter_matrix(2, 1) = 1.0;
                filter_matrix = static_cast<double>(1) / static_cast<double>(8) * filter_matrix;
                return  filter_matrix;

            case smoothing_av2:
                filter_matrix = static_cast<double>(1) / static_cast<double>(9) * Eigen::MatrixXd::Ones(3, 3);
                break;

            case sharpening_sh1:
                filter_matrix(0, 1) = -1.0;
                filter_matrix(1, 0) = -1.0;
                filter_matrix(1, 1) = 5.0;
                filter_matrix(1, 2) = -1.0;
                filter_matrix(2, 1) = -1.0;
                break;

            case sharpening_sh2:
                filter_matrix(0, 1) = -3.0;
                filter_matrix(1, 0) = -1.0;
                filter_matrix(1, 1) = 9.0;
                filter_matrix(1, 2) = -3.0;
                filter_matrix(2, 1) = -1.0;
                break;

            case sobel_vertical_ed1:
                filter_matrix(0, 0) = -1.0;
                filter_matrix(1, 0) = -2.0;
                filter_matrix(2, 0) = -1.0;
                filter_matrix(0, 2) = 1.0;
                filter_matrix(1, 2) = 2.0;
                filter_matrix(2, 2) = 1.0;
                break;

            case sobel_horizontal_ed2:
                filter_matrix(0, 0) = -1.0;
                filter_matrix(0, 1) = -2.0;
                filter_matrix(0, 2) = -1.0;
                filter_matrix(2, 0) = 1.0;
                filter_matrix(2, 1) = 2.0;
                filter_matrix(2, 2) = 1.0;
                break;

            case laplacian_edge_lap:
                filter_matrix(0, 1) = -1.0;
                filter_matrix(1, 0) = -1.0;
                filter_matrix(1, 1) = 4.0;
                filter_matrix(1, 2) = -1.0;
                filter_matrix(2, 1) = -1.0;
                break;

            default:
                throw std::invalid_argument("Unknown filter name");
        }
        return filter_matrix;
    }

}
