#ifndef MATRIX_UTILS
#define MATRIX_UTILS

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace matrix_utils {
    /**
     * Take an Eigen MatrixXd and return a sparse matrix with zero padding at the boundaries.
     * The sparse matrix structure is used to avoid wasting space.
     * @param matrix matrix to pad.
     * @return padded original matrix.
     */
    Eigen::SparseMatrix<float> zero_padding(const Eigen::MatrixXd &matrix);

    /**
     * Given the row and column, check if the tuple is within the indices of the matrix.
     * @param matrix matrix in which to check indices.
     * @param row row index.
     * @param col column index.
     * @return true if the index is inside the matrix, false otherwise.
     */
    bool isIndexOutOfBounds(const Eigen::MatrixXd& matrix, int row, int col);

    /**
     * A list of the most popular and well-known filters.
     */
    enum Filter: short;

    /**
     * Given a filter name, returns the matrix of the filter.
     * @param filter_name is the name of the filter to be returned.
     * @return the desired filter.
     * @throw std::invalid_argument if the filter name doesn't exist.
     */
    Eigen::MatrixXd create_filter(Filter filter_name);
}

#endif
