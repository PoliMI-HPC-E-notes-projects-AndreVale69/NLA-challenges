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
    bool is_index_out_of_bounds(const Eigen::MatrixXd& matrix, int row, int col);

    /**
     * Given a matrix, return true if it is symmetric.
     * @param matrix matrix to check that it is symmetrical.
     * @return true if the matrix is symmetric, false otherwise.
     */
    bool is_symmetric(const Eigen::SparseMatrix<double> &matrix);

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

    /**
     * Create a convolution matrix. It consists of (5*4) + 3*(cols-2)*2 + 3*(rows-2)*2 non-zero entries.
     * @param filter filter to be applied to the matrix; its values will be filled into the convolution matrix.
     * @param matrix matrix to which the convolution matrix is applied.
     * @return the convolution matrix as a sparse matrix.
     */
    Eigen::SparseMatrix<double> create_convolution_matrix(
        const Eigen::Matrix<double, 3, 3> &filter, const Eigen::MatrixXd &matrix
    );

    /**
     * Save an eigen vector to a mtx file.
     * @param filename filename where the mtx vector will be saved.
     * @param vector the vector to be stored.
     */
    void save_market_vector(const char * filename, const Eigen::VectorXd& vector);
}

#endif
