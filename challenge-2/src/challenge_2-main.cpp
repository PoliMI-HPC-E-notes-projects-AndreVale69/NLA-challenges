#ifdef __CYGWIN__
    std::cout << "The script has been tested on one of the Debian distributions, we don't guarantee that it will work."
#endif
#ifdef __APPLE__
    std::cout << "The script has been tested on one of the Debian distributions, we don't guarantee that it will work."
#endif


#include <filesystem>
#include <iostream>
#include <random>
#include <thread>
#include <unsupported/Eigen/SparseExtra>

#include "utils/matrix_utils.hpp"

#include "utils/image_manipulation.hpp"
#include "utils/matrix_utils.hpp"

using namespace std;
using namespace Eigen;
using namespace matrix_utils;


/**
 * This function is an override of the original loadMarketVector function.
 * The original has some problems, perhaps compatibility issues between versions of the libraries.
 * @return true if the image was successfully loaded, false otherwise.
 */
template<typename VectorType>
bool fixed_load_market_vector(VectorType& vec, const std::string& filename)
{
    typedef typename VectorType::Scalar Scalar;
    std::ifstream in(filename.c_str(), std::ios::in);
    if(!in)
        return false;

    std::string line;
    int n(0);
    do
    { // Skip comments
        std::getline(in, line); eigen_assert(in.good());
    } while (line[0] == '%');
    std::istringstream newline(line);
    newline  >> n;
    eigen_assert(n>0);
    vec.resize(n);
    int i = 0;
    Scalar value, index;
    while ( std::getline(in, line) && i < n ){
        std::istringstream newline(line);
        newline >> index;
        newline >> value;
        vec(i++) = value;
    }
    in.close();
    if (i!=n){
        std::cerr<< "Unable to read all elements from file " << filename << "\n";
        return false;
    }
    return true;
}



int main() {
    /**********
     * Task 1 *
     **********/
    // Load image from file
    int width = 0, height = 0, channels = 0, row_offset = 0;
    unsigned char* image_data;
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-2/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }

    static MatrixXd einstein_matrix(height, width),
                    einstein_matrix_transpose(height, width),
                    einstein_matrix_transpose_times_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            einstein_matrix(i, j) = static_cast<double>(image_data[row_offset + j]);
        }
    }
    einstein_matrix_transpose = einstein_matrix.transpose();
    einstein_matrix_transpose_times_matrix = einstein_matrix_transpose * einstein_matrix;

    printf(
    "\nTask 1. Load the image as an Eigen matrix with size m * n."
        "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0"
        "(black) and 255 (white)."
        "Compute the matrix product A^{T}A and report the Euclidean norm of A^{T}A.\n"
        "Answer: %f\n", einstein_matrix_transpose_times_matrix.norm()
    );


    /**********
     * Task 2 *
     **********/
    // SelfAdjointEigenSolver
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    const SelfAdjointEigenSolver<MatrixXd> eigensolver(einstein_matrix_transpose_times_matrix);
    // Check all is ok
    if (eigensolver.info() != Success) {
        abort();
    }
    // Take eigenvalues "the eigenvalues are sorted in increasing order"
    const VectorXd& eigenvalues = eigensolver.eigenvalues();
    const long eigenvalues_size = eigenvalues.size();
    printf(
        "\nTask 2. Solve the eigenvalue problem A^{T}Ax = lambda x using the proper solver "
        "provided by the Eigen library. Report the two largest computed singular values of A.\n"
        "Answer: %f (largest), %f (second largest)\n",
        sqrt(eigenvalues.coeff(eigenvalues_size-1)), sqrt(eigenvalues.coeff(eigenvalues_size-2))
    );


    /**********
     * Task 3 *
     **********/
    // Export using ATA name (A^T * A)
    saveMarket(einstein_matrix_transpose_times_matrix, "../challenge-2/resources/ATA.mtx");
    // Run LIS
    if (system("./../challenge-2/resources/run_lis.sh") != 0) {
        if (system("./../resources/run_lis.sh") != 0) {
            throw runtime_error("Error loading LIS modules");
        }
    }

    // Open results
    ifstream inputFile("../challenge-2/resources/result.txt");
    if (!inputFile.is_open()) {
        inputFile.open("../resources/result.txt");
        if (!inputFile.is_open()) {
            cerr << "Error opening the file!" << endl;
            return 1;
        }
    }

    // Save results
    unique_ptr<string> lis_result;
    {
        string line;
        for (int i = 0; getline(inputFile, line); ++i) {
            if (i == 13) {
                lis_result = make_unique<string>(line.erase(0, line.find_first_of("=")+2));
                break;
            }
        }
    }
    inputFile.close();

    printf(
        "\nTask 3. Export matrix A^{T}A in the matrix market format and move it to the lis-2.1.6/test folder. "
        "Using the proper iterative solver available in the LIS library compute the largest eigenvalue of A^{T}A "
        "up to a tolerance of 10^{-8}. Report the computed eigenvalue. "
        "Is the result in agreement with the one obtained in the previous point?"
        "\nAnswer: yes, the value obtained is %s\n", lis_result->c_str()
    );


    /**********
     * Task 4 *
     **********/
    // Run LIS
    if (system("./../challenge-2/resources/run_lis_shift.sh") != 0) {
        if (system("./../resources/run_lis_shift.sh") != 0) {
            throw runtime_error("Error loading LIS modules");
        }
    }

    // Open results
    inputFile.open("../challenge-2/resources/result_shift.txt");
    if (!inputFile.is_open()) {
        inputFile.open("../resources/result_shift.txt");
        if (!inputFile.is_open()) {
            cerr << "Error opening the file!" << endl;
            return 1;
        }
    }

    // Save results
    {
        string line;
        for (int i = 0; getline(inputFile, line); ++i) {
            if (i == 14) {
                lis_result = make_unique<string>(line.erase(0, line.find_first_of("=")+2));
                break;
            }
        }
    }
    inputFile.close();

    printf(
        "\nTask 4. Find a shift mu in R yielding an acceleration of the previous eigensolver. "
        "Report mu and the number of iterations required to achieve a tolerance of 10^{-8}."
        "\nAnswer: mu = 4.244525e+07 and %s iterations\n", lis_result->c_str()
    );


    /**********
     * Task 5 *
     **********/
    BDCSVD svd (einstein_matrix, ComputeThinU | ComputeThinV);
    VectorXd sigma = svd.singularValues();
    printf(
    "\nTask 5. Using the SVD module of the Eigen library, perform a singular value decomposition of the "
        "matrix A. Report the Euclidean norm of the diagonal matrix Sigma of the singular values."
        "\nAnswer: %f\n", sigma.norm()
    );


    /**********
     * Task 6 *
     **********/
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    MatrixXd C_40(U.rows(), 40), D_40(V.rows(), 40), C_80(U.rows(), 80), D_80(V.rows(), 80);
    int k = 40;
    for (int col = 0; col < k; ++col) {
        C_40.col(col) = U.col(col);
        D_40.col(col) = sigma[col] * V.col(col);
    }
    k = 80;
    for (int col = 0; col < k; ++col) {
        C_80.col(col) = U.col(col);
        D_80.col(col) = sigma[col] * V.col(col);
    }
    printf(
        "\nTask 6. Compute the matrices C and D described in (1) assuming k = 40 and k = 80. "
        "Report the number of nonzero entries in the matrices C and D."
        "\nAnswer: (k = 40, nnz(C) = %ld, nnz(D) = %ld), (k = 80, nnz(C) = %ld, nnz(D) = %ld)\n",
        C_40.nonZeros(), D_40.nonZeros(), C_80.nonZeros(), D_80.nonZeros()
    );


    /**********
     * Task 7 *
     **********/
    MatrixXd A_tilde_40 = (C_40 * D_40.transpose()).transpose();
    MatrixXd A_tilde_80 = (C_80 * D_80.transpose()).transpose();
    // Use Eigen's unaryExpr to map the grayscale values
    Matrix<unsigned char, Dynamic, Dynamic> compressed_image_40 = Matrix<unsigned char, Dynamic, Dynamic>(
        A_tilde_40.unaryExpr([](const double val) -> unsigned char {
            return static_cast<unsigned char>(val > 255.0 ? 255.0 : (val < 0 ? 0 : val));
        }));
    Matrix<unsigned char, Dynamic, Dynamic> compressed_image_80 = Matrix<unsigned char, Dynamic, Dynamic>(
        A_tilde_80.unaryExpr([](const double val) -> unsigned char {
            return static_cast<unsigned char>(val > 255.0 ? 255.0 : (val < 0 ? 0 : val));
    }));

    // Save the image using stbi_write_jpg
    const char* compressed_image_filename_40 = "compressed_image_k40.png";
    const char* clion_compressed_image_filename_40 = "../challenge-2/resources/compressed_image_k40.png";
    const char* compressed_image_filename_80 = "compressed_image_k80.png";
    const char* clion_compressed_image_filename_80 = "../challenge-2/resources/compressed_image_k80.png";
    // thread feature to speedup I/O operations
    std::thread compressed_image_save_40(
        image_manipulation::save_image_to_file,
        compressed_image_filename_40,
        A_tilde_40.rows(),
        A_tilde_40.cols(),
        1,
        compressed_image_40.data(),
        A_tilde_40.rows()
    );
    std::thread compressed_image_clion_save_40(
        image_manipulation::save_image_to_file,
        clion_compressed_image_filename_40,
        A_tilde_40.rows(),
        A_tilde_40.cols(),
        1,
        compressed_image_40.data(),
        A_tilde_40.rows()
    );
    std::thread compressed_image_save_80(
        image_manipulation::save_image_to_file,
        compressed_image_filename_80,
        A_tilde_80.rows(),
        A_tilde_80.cols(),
        1,
        compressed_image_80.data(),
        A_tilde_80.rows()
    );
    std::thread compressed_image_clion_save_80(
        image_manipulation::save_image_to_file,
        clion_compressed_image_filename_80,
        A_tilde_80.rows(),
        A_tilde_80.cols(),
        1,
        compressed_image_80.data(),
        A_tilde_80.rows()
    );
    compressed_image_save_40.join();
    compressed_image_save_80.join();
    compressed_image_clion_save_40.join();
    compressed_image_clion_save_80.join();
    printf(
        "\nTask 7. Compute the compressed images as the matrix product CD^{T} (again for k = 40 and k = 80). "
        "Export and upload the resulting images in .png."
        "\nAnswer: see the pictures %s\nAnd: %s\nAnd: %s\nAnd: %s\n",
        filesystem::absolute(compressed_image_filename_40).c_str(),
        filesystem::absolute(clion_compressed_image_filename_40).c_str(),
        filesystem::absolute(compressed_image_filename_80).c_str(),
        filesystem::absolute(clion_compressed_image_filename_80).c_str()
    );


    /**********
     * Task 8 *
     **********/
    SparseMatrix<double> chessboard(200, 200);
    create_chessboard(chessboard);
    printf(
        "\nTask 8. Using Eigen create a black and white checkerboard image "
        "with height and width equal to 200 pixels. "
        "Report the Euclidean norm of the matrix corresponding to the image."
        "\nAnswer: %f\n", chessboard.norm()
    );


    /**********
     * Task 9 *
     **********/
    // Create a random device and a Mersenne Twister random number generator
    // See more: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=285a65e11dbb6183a963489bc30b28ab04c6d7cf
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution d(-50, 50);

    // Create result matrix
    MatrixXd chessboard_noise_matrix(200, 200);
    double new_value = 0;

    // Fill the matrices with image data
    for (int i = 0; i < 200; ++i) {
        // row_offset = i * width;
        for (int j = 0; j < 200; ++j) {
            new_value = chessboard.coeff(i, j) + d(gen);
            chessboard_noise_matrix(i, j) = new_value > 255.0 ? 255.0 : (new_value < 0 ? 0 : new_value);
        }
    }

    // Use Eigen's unaryExpr to map the grayscale values
    unique_ptr<Matrix<unsigned char, Dynamic, Dynamic, RowMajor>> chessboard_noise = make_unique
    <Matrix<unsigned char, Dynamic, Dynamic, RowMajor>>(
        chessboard_noise_matrix.unaryExpr([](const double val) -> unsigned char {
            return static_cast<unsigned char>(val);
        })
    );

    // Save the image using stbi_write_jpg
    const char* noise_filename = "noise.png";
    const char* clion_noise_filename = "../challenge-2/resources/noise.png";
    // thread feature to speedup I/O operations
    std::thread noise_save(
        image_manipulation::save_image_to_file,
        noise_filename, 200, 200, 1, chessboard_noise->data(), 200
    );
    std::thread noise_clion_save(
        image_manipulation::save_image_to_file,
        clion_noise_filename, 200, 200, 1, chessboard_noise->data(), 200
    );
    noise_save.join();
    noise_clion_save.join();
    chessboard_noise.reset();

    printf(
        "\nTask 9. Introduce a noise into the checkerboard image by adding random fluctuations "
        "of color ranging between [-50, 50] to each pixel. "
        "Export the resulting image in .png and upload it."
        "\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(noise_filename).c_str(),
        filesystem::absolute(clion_noise_filename).c_str()
    );


    /***********
     * Task 10 *
     ***********/
    BDCSVD svd_noise (chessboard_noise_matrix, ComputeThinU | ComputeThinV);
    VectorXd sigma_noise = svd_noise.singularValues();
    printf(
        "\nTask 10. Using the SVD module of the Eigen library, perform a singular value decomposition of the "
        "matrix corresponding to the noisy image."
        "Report the two largest computed singular values."
        "\nAnswer: %f (largest), %f (second largest)\n",
        sigma_noise.coeff(0), sigma_noise.coeff(1)
    );


    /***********
     * Task 11 *
     ***********/
    MatrixXd U_noise = svd_noise.matrixU();
    MatrixXd V_noise = svd_noise.matrixV();
    MatrixXd C_5(U_noise.rows(), 5), D_5(V_noise.rows(), 5),
             C_10(U_noise.rows(), 10), D_10(V_noise.rows(), 10);
    k = 5;
    for (int col = 0; col < k; ++col) {
        C_5.col(col) = U_noise.col(col);
        D_5.col(col) = sigma_noise[col] * V_noise.col(col);
    }
    k = 10;
    for (int col = 0; col < k; ++col) {
        C_10.col(col) = U_noise.col(col);
        D_10.col(col) = sigma_noise[col] * V_noise.col(col);
    }
    printf(
        "\nTask 11. Starting from the previously computed SVD, creates the matrices C and D defined in (1)"
        " assuming k = 5 and k = 10. Report the size of the matrices C and D."
        "\nAnswer: (k = 5, C = (%ld rows, %ld cols), D = (%ld rows, %ld cols)), (k = 10, C = (%ld rows, %ld cols), D = (%ld rows, %ld cols))\n",
        C_5.rows(), C_5.cols(),
        D_5.rows(), D_5.cols(),
        C_10.rows(), C_10.cols(),
        D_10.rows(), D_10.cols()
    );


    /**********
     * Task 12 *
     **********/
    MatrixXd A_tilde_5 = (C_5 * D_5.transpose()).transpose();
    MatrixXd A_tilde_10 = (C_10 * D_10.transpose()).transpose();
    // Use Eigen's unaryExpr to map the grayscale values
    Matrix<unsigned char, Dynamic, Dynamic> compressed_image_5 = Matrix<unsigned char, Dynamic, Dynamic>(
        A_tilde_5.unaryExpr([](const double val) -> unsigned char {
            return static_cast<unsigned char>(val > 255.0 ? 255.0 : (val < 0 ? 0 : val));
        }));
    Matrix<unsigned char, Dynamic, Dynamic> compressed_image_10 = Matrix<unsigned char, Dynamic, Dynamic>(
        A_tilde_10.unaryExpr([](const double val) -> unsigned char {
            return static_cast<unsigned char>(val > 255.0 ? 255.0 : (val < 0 ? 0 : val));
    }));

    // Save the image using stbi_write_jpg
    const char* compressed_image_filename_5 = "compressed_noise_image_k5.png";
    const char* clion_compressed_image_filename_5 = "../challenge-2/resources/compressed_noise_image_k5.png";
    const char* compressed_image_filename_10 = "compressed_noise_image_k10.png";
    const char* clion_compressed_image_filename_10 = "../challenge-2/resources/compressed_noise_image_k10.png";
    // thread feature to speedup I/O operations
    std::thread compressed_image_save_5(
        image_manipulation::save_image_to_file,
        compressed_image_filename_5,
        A_tilde_5.rows(),
        A_tilde_5.cols(),
        1,
        compressed_image_5.data(),
        A_tilde_5.rows()
    );
    std::thread compressed_image_clion_save_5(
        image_manipulation::save_image_to_file,
        clion_compressed_image_filename_5,
        A_tilde_5.rows(),
        A_tilde_5.cols(),
        1,
        compressed_image_5.data(),
        A_tilde_5.rows()
    );
    std::thread compressed_image_save_10(
        image_manipulation::save_image_to_file,
        compressed_image_filename_10,
        A_tilde_10.rows(),
        A_tilde_10.cols(),
        1,
        compressed_image_10.data(),
        A_tilde_10.rows()
    );
    std::thread compressed_image_clion_save_10(
        image_manipulation::save_image_to_file,
        clion_compressed_image_filename_10,
        A_tilde_10.rows(),
        A_tilde_10.cols(),
        1,
        compressed_image_10.data(),
        A_tilde_10.rows()
    );
    compressed_image_save_5.join();
    compressed_image_save_10.join();
    compressed_image_clion_save_5.join();
    compressed_image_clion_save_10.join();
    printf(
        "\nTask 12. Compute the compressed images as the matrix product CD^{T} (again for k = 5 and k = 10). "
        "Export and upload the resulting images in .png."
        "\nAnswer: see the pictures %s\nAnd: %s\nAnd: %s\nAnd: %s\n",
        filesystem::absolute(compressed_image_filename_5).c_str(),
        filesystem::absolute(clion_compressed_image_filename_5).c_str(),
        filesystem::absolute(compressed_image_filename_10).c_str(),
        filesystem::absolute(clion_compressed_image_filename_10).c_str()
    );

    /***********
     * Task 13 *
     ***********/
    printf(
        "\nTask 13. A low k value means higher compression because fewer terms are used. "
        "This is clearly seen if we look at the image of Einstein with k = 40 compared to the image with k = 80; "
        "the first image has lost a lot of details and the image is not of good quality; "
        "in contrast, the second image has more details because it contains more values (k is larger). "
        "Another interesting result is that a low value of k also reduces the noise. "
        "This can be seen in the chessboard image, where the k = 5 image has less noise than the k = 10 image. "
        "So what do these tests tell us? "
        "Well, that the truncated svd method needs to be used carefully so that we can use it to balance "
        "data compression and detail preservation. Obviously, we can't use k too high "
        "(+ details, noise doesn't change much) or k too low (+ compression, reduces noise, - details). "
        "However, low k-values work well (in terms of noise reduction) for images with little detail, "
        "such as a chessboard image, but k-values reduce the details too much for an image with a lot of detail "
        "(Einstein image)."
    );
    return 0;
}
