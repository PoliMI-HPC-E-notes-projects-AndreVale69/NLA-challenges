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
#include <cstdlib>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "utils/image_manipulation.hpp"
#include "utils/matrix_utils.hpp"

using namespace std;
using namespace Eigen;
using namespace matrix_utils;

// TODO: tmp workaround to include the enum
enum Filter: short {
    smoothing_av1,
    smoothing_av2,
    sharpening_sh1,
    sharpening_sh2,
    sobel_vertical_ed1,
    sobel_horizontal_ed2,
    laplacian_edge_lap
};


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
    unsigned char* image_data, * noise_image_data;
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_img(height, width);
    static MatrixXd dark_einstein_img(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            dark_einstein_img(i, j) = static_cast<double>(image_data[row_offset + j]);
        }
    }
    einstein_img = dark_einstein_img.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    printf(
        "\nTask 1. Load the image as an Eigen matrix with size m*n. "
        "Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere "
        "between 0 (black) and 255 (white). Report the size of the matrix.\n"
        "Answer: %ld * %ld (rows * cols)\n", einstein_img.rows(), einstein_img.cols()
    );


    /**********
     * Task 2 *
     **********/
    // Create a random device and a Mersenne Twister random number generator
    // See more: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=285a65e11dbb6183a963489bc30b28ab04c6d7cf
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution d(-50, 50);

    // Create result matrix
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> einstein_noise(height, width);
    MatrixXd dark_noise_img(height, width);
    int new_value = 0;

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            new_value = image_data[row_offset + j] + d(gen);
            dark_noise_img(i, j) = new_value > 255.0 ? 255.0 : (new_value < 0 ? 0 : new_value);
        }
    }

    // Use Eigen's unaryExpr to map the grayscale values
    einstein_noise = dark_noise_img.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });

    // Save the image using stbi_write_jpg
    const char* noise_filename = "noise.png";
    const char* clion_noise_filename = "../challenge-1/resources/noise.png";
    // thread feature to speedup I/O operations
    std::thread noise_save(
        image_manipulation::save_image_to_file,
        noise_filename, width, height, 1, einstein_noise.data(), width
    );
    std::thread noise_clion_save(
        image_manipulation::save_image_to_file,
        clion_noise_filename, width, height, 1, einstein_noise.data(), width
    );
    noise_save.join();
    noise_clion_save.join();

    printf(
        "\nTask 2. Introduce a noise signal into the loaded image "
        "by adding random fluctuations of color ranging between [-50, 50] to each pixel. "
        "Export the resulting image in .png and upload it.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(noise_filename).c_str(),
        filesystem::absolute(clion_noise_filename).c_str()
    );


    /**********
     * Task 3 *
     **********/
    // Create original image as v vector
    const long size_vector_einstein_img = dark_einstein_img.cols() * dark_einstein_img.rows();
    VectorXd v(size_vector_einstein_img);
    try {
        image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        image_data = image_manipulation::load_image_from_file(
            "../resources/Albert_Einstein_Head.jpg", width, height, channels
        );
    }
    for (int i = 0; i < size_vector_einstein_img; ++i) {
        v(i) = static_cast<double>(image_data[i]);
    }
    // Load noise image and create noise image as w vector
    try {
        noise_image_data = image_manipulation::load_image_from_file(
            "../challenge-1/resources/noise.png", width, height, channels
        );
    } catch ([[maybe_unused]] const std::runtime_error& e) {
        noise_image_data = image_manipulation::load_image_from_file(
            "../resources/noise.png", width, height, channels
        );
    }
    const long size_vector_noise_img = dark_noise_img.cols() * dark_noise_img.rows();
    VectorXd w(size_vector_noise_img);
    for (int i = 0; i < size_vector_noise_img; ++i) {
        w(i) = static_cast<double>(noise_image_data[i]);
    }

    // Verify that each vector has m*n components
    if (v.size() != size_vector_einstein_img || w.size() != size_vector_noise_img) {
        stringstream error;
        error << "Convertion from matrix to vector failed.\n" <<
            "Vector v size: " << v.size() << ", expected size: " << size_vector_einstein_img << '\n' <<
            "Vector w size: " << w.size() << ", expected size: " << size_vector_noise_img << '\n';
        throw runtime_error(error.str());
    }

    printf(
        "\nTask 3. Reshape the original and noisy images as vectors v and w, respectively. "
        "Verify that each vector has m n components. Report here the Euclidean norm of v.\nAnswer: %f\n", v.norm()
    );


    /**********
     * Task 4 *
     **********/
    // Create smoothing filter H_av2
    MatrixXd H_av2 = create_filter(static_cast<matrix_utils::Filter>(smoothing_av2));
    // Create convolution matrix
    SparseMatrix<double> A1 = create_convolution_matrix(H_av2, dark_einstein_img);
    printf("\nTask 4.Write the convolution operation corresponding to the smoothing kernel H_{av2} "
           "as a matrix vector multiplication between a matrix A_{1} having size mn times mn and the image vector."
           "Report the number of non-zero entries in A_{1}.\nAnswer: %ld\n", A1.nonZeros());


    /**********
     * Task 5 *
     **********/
    // Convolution using the matrix multiplication technique
    VectorXd convolution_multiplication = A1*w;
    static MatrixXd smoothing_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            smoothing_matrix(i, j) = convolution_multiplication[row_offset + j];
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> result(height, width);
    result = smoothing_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* smoothing_filename = "smoothing.png";
    const char* clion_smoothing_filename = "../challenge-1/resources/smoothing.png";
    std::thread smoothing_save(
        image_manipulation::save_image_to_file,
        smoothing_filename, width, height, 1, result.data(), width
    );
    std::thread smoothing_clion_save(
        image_manipulation::save_image_to_file,
        clion_smoothing_filename, width, height, 1, result.data(), width
    );
    smoothing_save.join();
    smoothing_clion_save.join();

    printf("\nTask 5. Apply the previous smoothing filter to the noisy image by performing "
        "the matrix vector multiplication A_{1}w Export the resulting image.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(smoothing_filename).c_str(),
        filesystem::absolute(clion_smoothing_filename).c_str());


    /**********
     * Task 6 *
     **********/
    // Create smoothing filter H_av2
    MatrixXd H_sh2 = create_filter(static_cast<matrix_utils::Filter>(sharpening_sh2));
    // Create convolution matrix
    SparseMatrix<double> A2 = create_convolution_matrix(H_sh2, dark_einstein_img);
    // According to the symmetric definition of a matrix: A = A^{T} iff A is symmetric
    printf("\nTask 6. Write the convolution operation corresponding to the sharpening kernel H_{sh2} "
           "as a matrix vector multiplication by a matrix A_{2} having size mn * mn. "
           "Report the number of non-zero entries in A_{2}. Is A_{2} symmetric?\nAnswer: %ld, is A_{2} symmetric? %s\n",
           A2.nonZeros(), is_symmetric(A2) ? "true" : "false");


    /**********
     * Task 7 *
     **********/
    // Convolution using the matrix multiplication technique
    VectorXd sharpening_convolution_matrix = A2*v;
    static MatrixXd sharpening_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            const double val = sharpening_convolution_matrix[row_offset + j];
            sharpening_matrix(i, j) = val > 255.0 ? 255.0 : (val < 0 ? 0 : val);
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpening_result(height, width);
    sharpening_result = sharpening_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* sharpening_filename = "sharpening.png";
    const char* clion_sharpening_filename = "../challenge-1/resources/sharpening.png";
    std::thread sharpening_save(
        image_manipulation::save_image_to_file,
        sharpening_filename, width, height, 1, sharpening_result.data(), width
    );
    std::thread sharpening_clion_save(
        image_manipulation::save_image_to_file,
        clion_sharpening_filename, width, height, 1, sharpening_result.data(), width
    );
    sharpening_save.join();
    sharpening_clion_save.join();

    printf(
        "\nTask 7. Apply the previous sharpening filter to the original image by performing the matrix "
        "vector multiplication A_{2}v. Export the resulting image.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(sharpening_filename).c_str(),
        filesystem::absolute(clion_sharpening_filename).c_str()
    );


    /**********
     * Task 8 *
     **********/
    // Export
    saveMarket(A2, "../challenge-1/resources/A2.mtx");
    save_market_vector("../challenge-1/resources/w.mtx", w);

    // Run LIS
    if (system("./../challenge-1/resources/run_lis.sh") != 0) {
        if (system("./../resources/run_lis.sh") != 0) {
            throw runtime_error("Error loading LIS modules");
        }
    }

    // Open results
    ifstream inputFile("../challenge-1/resources/result.txt");
    if (!inputFile.is_open()) {
        inputFile.open("../resources/result.txt");
        if (!inputFile.is_open()) {
            cerr << "Error opening the file!" << endl;
            return 1;
        }
    }

    // Save results
    string n_iterations, final_residual, line;
    for (int i = 0; getline(inputFile, line); ++i) {
        if (i == 12) {
            n_iterations = line.erase(0, line.find_first_of("=")+2);
        } else if (i == 19) {
            final_residual = line.erase(0, line.find_first_of("=")+2);
        }
    }
    inputFile.close();

    printf(
        "\nTask 8. Export the Eigen matrix $A_{2}$ and vector $w$ in the .mtx format."
        " Using a suitable iterative solver and preconditioner technique available in the LIS library compute "
        "the approximate solution to the linear system A_{2}x = w prescribing a tolerance of 10^{-9}. "
        "Report here the iteration count and the final residual."
        "\nAnswer: number of iterations %s, number of final residual %s\n", n_iterations.c_str(), final_residual.c_str()
    );


    /**********
     * Task 9 *
     **********/
    VectorXd mat;
    fixed_load_market_vector(mat, "../challenge-1/resources/sol.mtx");
    static MatrixXd solution_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            solution_matrix(i, j) = mat[row_offset + j];
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> solution_result(height, width);
    solution_result = solution_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* solution_filename = "lis_solution.png";
    const char* clion_solution_filename = "../challenge-1/resources/lis_solution.png";
    std::thread result_thread(
        image_manipulation::save_image_to_file,
        solution_filename, width, height, 1, solution_result.data(), width
    );
    std::thread clion_result_thread(
        image_manipulation::save_image_to_file,
        clion_solution_filename, width, height, 1, solution_result.data(), width
    );
    result_thread.join();
    clion_result_thread.join();

    printf("\nTask 9. Import the previous approximate solution vector x in Eigen "
        "and then convert it into a .png image. Upload the resulting file here.\nAnswer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(solution_filename).c_str(),
        filesystem::absolute(clion_solution_filename).c_str());


    /***********
     * Task 10 *
     ***********/
    // Create detection kernel H_lap
    MatrixXd H_lap = create_filter(static_cast<matrix_utils::Filter>(laplacian_edge_lap));
    // Create convolution matrix
    SparseMatrix<double> A3 = create_convolution_matrix(H_lap, dark_einstein_img);
    printf("\nTask 10. Write the convolution operation corresponding to the detection kernel H_{lap} "
           "as a matrix vector multiplication by a matrix A_{3} having size mn times mn. "
           "Is matrix A_{3} symmetric?\nAnswer: %s\n", is_symmetric(A3) ? "true" : "false");


    /***********
     * Task 11 *
     ***********/
    // Convolution using the matrix multiplication technique
    VectorXd edge_detection_convolution_matrix = A3*v;
    static MatrixXd edge_detection_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            const double val = edge_detection_convolution_matrix[row_offset + j];
            edge_detection_matrix(i, j) = val > 255.0 ? 255.0 : (val < 0 ? 0 : val);
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> edge_detection_result(height, width);
    edge_detection_result = edge_detection_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* edge_detection_filename = "edge_detection.png";
    const char* clion_edge_detection_filename = "../challenge-1/resources/edge_detection.png";
    std::thread edge_detection_save(
        image_manipulation::save_image_to_file,
        edge_detection_filename, width, height, 1, edge_detection_result.data(), width
    );
    std::thread edge_detection_clion_save(
        image_manipulation::save_image_to_file,
        clion_edge_detection_filename, width, height, 1, edge_detection_result.data(), width
    );
    edge_detection_save.join();
    edge_detection_clion_save.join();

    printf("\nTask 11. Apply the previous edge detection filter to the original image by performing "
           "the matrix vector multiplication A_{3}v. Export and upload the resulting image."
           "\nAnswer: see the figure %s\nAnd: %s\n",
           filesystem::absolute(edge_detection_filename).c_str(),
           filesystem::absolute(clion_edge_detection_filename).c_str());


    /***********
     * Task 12 *
     ***********/
    // Create identity matrix
    SparseMatrix<double> identity(A3.rows(), A3.cols());
    identity.setIdentity();
    // Store I_A3 using RowMajor to gain performance using OpenMP (fopenmp compilation flag required obv)
    SparseMatrix<double> I_A3 = identity + A3;
    SparseVector<double> y(A3.size());
    // Set parameters for solver
    // Convergence tolerance
    double tol = 1.e-10;
    // Maximum iterations
    int maxit = 100;
    BiCGSTAB<SparseMatrix<double>> bi_cgstab;
    bi_cgstab.setMaxIterations(maxit);
    bi_cgstab.setTolerance(tol);
    bi_cgstab.compute(I_A3);

    // Convert from dense to sparse
    SparseMatrix<double> w_sparse_vector(size_vector_noise_img, 1);
    vector<Triplet<double>> triplets;
    triplets.reserve(size_vector_einstein_img);
    for (int i = 0; i < size_vector_noise_img; ++i) {
        triplets.emplace_back(i, 0, w[i]);
    }
    w_sparse_vector.setFromTriplets(triplets.begin(), triplets.end());

    // Solve
    y = bi_cgstab.solve(w_sparse_vector);

    printf("\nTask 12. Using a suitable iterative solver available in the Eigen library compute the approximate solution"
    "of the linear system (I+A_{3})y = w, where I denotes the identity matrix, prescribing a tolerance of 10^{-10}."
    "Report here the iteration count and the final residual.\n"
    "Answer: the iteration count is %ld and the final residual is ", bi_cgstab.iterations());
    cout << bi_cgstab.error() << " (using BiCGSTAB).\n";


    /***********
     * Task 13 *
     ***********/
    static MatrixXd eigen_solution_matrix(height, width);
    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        row_offset = i * width;
        for (int j = 0; j < width; ++j) {
            double val = y.coeff(row_offset+j);
            eigen_solution_matrix(i, j) = val > 255.0 ? 255.0 : (val < 0 ? 0 : val);
        }
    }
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> eigen_solution(height, width);
    eigen_solution = eigen_solution_matrix.unaryExpr([](const double val) -> unsigned char {
      return static_cast<unsigned char>(val);
    });
    // thread feature to speedup I/O operations
    const char* eigen_solution_filename = "eigen_solution.png";
    const char* clion_eigen_solution_filename = "../challenge-1/resources/eigen_solution.png";
    std::thread eigen_solution_save(
        image_manipulation::save_image_to_file,
        eigen_solution_filename, width, height, 1, eigen_solution.data(), width
    );
    std::thread eigen_solution_clion_save(
        image_manipulation::save_image_to_file,
        clion_eigen_solution_filename, width, height, 1, eigen_solution.data(), width
    );
    eigen_solution_save.join();
    eigen_solution_clion_save.join();

    printf("\nTask 13. Convert the image stored in the vector y into a .png image and upload it."
        "Answer: see the figure %s\nAnd: %s\n",
        filesystem::absolute(eigen_solution_filename).c_str(),
        filesystem::absolute(clion_eigen_solution_filename).c_str());

    return 0;
}
