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

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_img(height, width);
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
        "\nTask 2. Solve the eigenvalue problem A^{T}Ax = lambda x using the proper solver"
        "provided by the Eigen library. Report the two largest computed singular values of A.\n"
        "Answer: %f (largest), %f (second largest)\n",
        eigenvalues.coeff(eigenvalues_size-1), eigenvalues.coeff(eigenvalues_size-2)
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
    string largest_eigenvalue, line;
    for (int i = 0; getline(inputFile, line); ++i) {
        if (i == 13) {
            largest_eigenvalue = line.erase(0, line.find_first_of("=")+2);
            break;
        }
    }
    inputFile.close();

    printf(
        "\nTask 3. Export matrix A^{T}A in the matrix market format and move it to the lis-2.1.6/test folder. "
        "Using the proper iterative solver available in the LIS library compute the largest eigenvalue of A^{T}A "
        "up to a tolerance of 10^{-8}. Report the computed eigenvalue. "
        "Is the result in agreement with the one obtained in the previous point?"
        "\nAnswer: yes, the value obtained is %s\n", largest_eigenvalue.c_str()
    );


    /**********
     * Task 8 *
     **********/
    MatrixXd chessboard = MatrixXd::Zero(200, 200);
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
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> chessboard_noise;
    MatrixXd chessboard_noise_matrix(200, 200);
    double new_value = 0;

    // Fill the matrices with image data
    for (int i = 0; i < 200; ++i) {
        // row_offset = i * width;
        for (int j = 0; j < 200; ++j) {
            new_value = chessboard(i, j) + d(gen);
            chessboard_noise_matrix(i, j) = new_value > 255.0 ? 255.0 : (new_value < 0 ? 0 : new_value);
        }
    }

    // Use Eigen's unaryExpr to map the grayscale values
    chessboard_noise = chessboard_noise_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Save the image using stbi_write_jpg
    const char* noise_filename = "noise.png";
    const char* clion_noise_filename = "../challenge-2/resources/noise.png";
    // thread feature to speedup I/O operations
    std::thread noise_save(
        image_manipulation::save_image_to_file,
        noise_filename, 200, 200, 1, chessboard_noise.data(), 200
    );
    std::thread noise_clion_save(
        image_manipulation::save_image_to_file,
        clion_noise_filename, 200, 200, 1, chessboard_noise.data(), 200
    );
    noise_save.join();
    noise_clion_save.join();

    printf(
        "\nTask 9. Introduce a noise into the checkerboard image by adding random fluctuations "
        "of color ranging between [-50, 50] to each pixel. "
        "Export the resulting image in .png and upload it."
        "\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(noise_filename).c_str(),
        filesystem::absolute(clion_noise_filename).c_str()
    );
    return 0;
}
