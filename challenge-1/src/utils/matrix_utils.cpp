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

    Eigen::SparseMatrix<double> create_convolution_matrix(
        const Eigen::Matrix<double, 3, 3> &filter, const Eigen::MatrixXd &matrix
    ) {
        // flag used to understand when to apply rows_offset value
        bool almost_one_col_valid = false;
        // rows_filled: the number of filled rows, in the end it should be equal
        //              to the number of rows of the convolution matrix.
        // rows_offset: an offset used to skip n zeros when the algorithm moves to the next row of the filter;
        //              so the algorithm considers the first row of the filter value,
        //              so rows_offset skips zeros, considers the middle values of the filter, and so on.
        // offset_filter: a counter that is updated each time the algorithm evaluates the neighbours;
        //                it is used to access the filter array; it allows to avoid the double index access (i,j).
        // upper_entries_to_skip: the number of entries to skip;
        //                        it allows to create a Toeplitz matrix without worrying about
        //                        the exact position in the original array; it is applied from the number of row 2.
        // four variables are used to define the boundaries of neighbourhood research:
        //      - row_lower_bound_neighbour
        //      - row_upper_bound_neighbour
        //      - col_left_bound_neighbour
        //      - col_right_bound_neighbour
        // they are used in the for statement
        int rows_filled = 0, rows_offset = 0, offset_filter = 0, upper_entries_to_skip = 0,
            row_lower_bound_neighbour = 0, row_upper_bound_neighbour = 0,
            col_left_bound_neighbour = 0, col_right_bound_neighbour = 0;
        // const allocation of rows and cols of the original matrix, and matrix size;
        // this avoids multiple memory accesses
        const int matrix_rows = static_cast<int>(matrix.rows());
        const int matrix_cols = static_cast<int>(matrix.cols());
        const long matrix_size = matrix.size();
        // use small array to take the values of the filter;
        // the size is so small that accessing an element n is very fast (O(n))
        // and negligible (in terms of performance)
        const double * filter_array = filter.data();
        // create the convolution (sparse) matrix and the triplet
        Eigen::SparseMatrix<double> convolution_matrix(matrix_size, matrix_size);
        std::vector<Eigen::Triplet<double>> triplet_list;
        triplet_list.reserve(filter.size() * matrix_size);

        // for each row of the matrix
        for (int row = 0; row < matrix_rows; ++row) {
            row_upper_bound_neighbour = row - 1;
            row_lower_bound_neighbour = row + 1;
            upper_entries_to_skip = (row - 1) * matrix_cols;
            // for each column of the matrix
            for (int col = 0; col < matrix_cols; ++col, ++rows_filled) {
                offset_filter = 0;
                rows_offset = 0;
                /**
                 * check the neighbours:
                 * x x x
                 * x o x
                 * x x x
                 * where o is the centre of the filter and the x's are its neighbours;
                 * set the new column boundaries before the check
                 */
                col_left_bound_neighbour = col-1;
                col_right_bound_neighbour = col+1;
                for (int i_row = row_upper_bound_neighbour; i_row <= row_lower_bound_neighbour; ++i_row) {
                    // reset the flag
                    almost_one_col_valid = false;
                    for (int j_col = col_left_bound_neighbour; j_col <= col_right_bound_neighbour; ++j_col, ++offset_filter) {
                        // check that the index neighbour is valid;
                        // this is essential if the filter is applied to the edges of the matrix
                        if (isIndexOutOfBounds(matrix, i_row, j_col)) {
                            continue;
                        }
                        // update the flag, the rows_offset will be updated
                        almost_one_col_valid = true;
                        // optimization to avoid garbage (zero) values; add iff > 0;
                        // use some available memory to store zeros;
                        // this should increase speed, but is it really necessary?
                        if (const auto filter_value = filter_array[offset_filter]; filter_value > 0.0) {
                            // if there is almost one row to skip, use the upper_entries_to_skip
                            if (row >= 2) {
                                triplet_list.emplace_back(
                                    rows_filled, j_col+rows_offset+upper_entries_to_skip, filter_value
                                );
                                continue;
                            }
                            triplet_list.emplace_back(rows_filled, j_col+rows_offset, filter_value);
                        }
                    }
                    // if almost a column has been evaluated, update the rows_offset counter
                    if (almost_one_col_valid) {
                        rows_offset += matrix_cols;
                    }
                }
            }
        }
        assert(rows_filled == matrix_size);
        // create the new sparse matrix using the triplet;
        // hey garbage collector, the triplet will be all yours soon!
        convolution_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
        return convolution_matrix;
    }

}
