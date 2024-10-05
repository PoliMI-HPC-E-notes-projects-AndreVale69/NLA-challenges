#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <cstdlib>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Dynamic;
/*
using SpMat = Eigen::SparseMatrix<double,Eigen::RowMajor>;
using SpVec = Eigen::VectorXd;
*/

int main(int argc, char** argv)
{
    // TASK 1
    const char * image_path = "./256px-Albert_Einstein_Head.jpeg";
    int width, height, channels;

    unsigned char* image_data = stbi_load(image_path, &width, &height, &channels, 1);
    if (!image_data) {
        cerr << "Error loading image!" << endl;
        return 1;
    }

    std::cout << "Matrix size: " << height << "x" << width << std::endl;
    
    double random_number;
    double value;
    int index;
    int i,j,row_offset;
    MatrixXd original(height, width), noisy(height, width);
    
    for (i = 0; i < height; i++) {
        row_offset = i *width;
        for (j = 0; j < width; j++) {
            random_number = (std::rand() % 101) - 50;
            value = std::max(static_cast<double>(image_data[row_offset+j]) + random_number, 0.0);
            value = std::min(value, 255.0);
            original(i, j) = static_cast<double>(image_data[row_offset+j]);
            noisy(i, j) = value;
        }
    }
    
    stbi_image_free(image_data);
    /*
    
    Matrix<unsigned char, Dynamic, Dynamic,Eigen::RowMajor> original_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    original_image = original.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });

    Matrix<unsigned char, Dynamic, Dynamic, Eigen::RowMajor> noisy_image(height, width);

    noisy_image = noisy.unaryExpr([](double val) -> unsigned char{
        return static_cast<unsigned char> (val*255.0);
    });

    */
    Matrix<unsigned char, Dynamic, Dynamic,Eigen::RowMajor> original_image(height, width);
    // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
    original_image = original.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val);
    });

    Matrix<unsigned char, Dynamic, Dynamic, Eigen::RowMajor> noisy_image(height, width);

    noisy_image = noisy.unaryExpr([](double val) -> unsigned char{
        return static_cast<unsigned char> (val);
    });

    const std::string output_image_path1 = "noisy_image.png";
    // Save the noisy image
    if (stbi_write_png(output_image_path1.c_str(), width, height, 1, noisy_image.data(), width) == 0) {
        std::cerr << "Error: Could not save output image" << std::endl;
        return 1;
    }
    std::cout << "Noisy image saved as " << output_image_path1 << std::endl;

    // TASK 2
    Eigen::VectorXd v(height*width);
    Eigen::VectorXd w(height*width);

    // Vettorializzare original_image (normalizzato tra 0 e 1)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            v(i * width + j) = original(i, j);  // Copia in v (già normalizzato)
        }
    }

    // Vettorializzare noisy_image (normalizzato tra 0 e 1)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            w(i * width + j) = noisy(i, j);  // Copia in w (già normalizzato)
        }
    }
    
    std::cout<<"mxn=" <<height<<"x" << width<< "=" <<height*width<<", v size:"<<v.size() << ", w size:"<<w.size()<<std::endl;
    std::cout<<"norm of v:"<<v.norm()<<std::endl;

    //TASK 4
    Eigen::SparseMatrix<double,Eigen::RowMajor> A1(v.size(),v.size());
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(v.size());
    
    for(i=0;i<v.size();i++){
        // Controllo per l'indice i-4
        if (i - 4 >= 0) {
            tripletList.push_back(Eigen::Triplet<double>(i, i - 4, 1.0 / 9.0));
        }
        // Controllo per l'indice i-3
        if (i - 3 >= 0) {
            tripletList.push_back(Eigen::Triplet<double>(i, i - 3, 1.0 / 9.0));
        }
        // Controllo per l'indice i-2
        if (i - 2 >= 0) {
            tripletList.push_back(Eigen::Triplet<double>(i, i - 2, 1.0 / 9.0));
        }
        // Controllo per l'indice i-1
        if (i - 1 >= 0) {
            tripletList.push_back(Eigen::Triplet<double>(i, i - 1, 1.0 / 9.0));
        }
        // Indice i
        tripletList.push_back(Eigen::Triplet<double>(i, i, 1.1 / 9.0));
        // Controllo per l'indice i+1
        if (i + 1 < v.size()) {
            tripletList.push_back(Eigen::Triplet<double>(i, i + 1, 1.0 / 9.0));
        }
        // Controllo per l'indice i+2
        if (i + 2 < v.size()) {
            tripletList.push_back(Eigen::Triplet<double>(i, i + 2, 1.0 / 9.0));
        }
        // Controllo per l'indice i+3
        if (i + 3 < v.size()) {
            tripletList.push_back(Eigen::Triplet<double>(i, i + 3, 1.0 / 9.0));
        }
        // Controllo per l'indice i+4
        if (i + 4 < v.size()) {
            tripletList.push_back(Eigen::Triplet<double>(i, i + 4, 1.0 / 9.0));
        }
    }
    A1.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout<<"A1 nonzeros :" << A1.nonZeros()<<std::endl;


    Eigen::VectorXd t4 = A1*v;//task 4


    // TASK 5 
    Eigen::VectorXd t5 = A1*w;
    MatrixXd t5matrix(height,width);
    

    for(i=0;i<height;i++){
        for(j=0;j<width;j++){
            t5matrix(i,j)= t5(i*width+j);
        }
    }
    
    Matrix<unsigned char, Dynamic, Dynamic, Eigen::RowMajor> t5image(height, width);

    t5image = t5matrix.unaryExpr([](double val) -> unsigned char{
        return static_cast<unsigned char> (val);
    });

    const std::string output_image_patht5 = "t5image.png";
    // Save the noisy image
    if (stbi_write_png(output_image_patht5.c_str(), width, height, 1, t5image.data(), width) == 0) {
        std::cerr << "Error: Could not save output image" << std::endl;
        return 1;
    }

    //TASK 6-7
    Eigen::SparseMatrix<double,Eigen::RowMajor> A2(v.size(),v.size());
    std::vector<Eigen::Triplet<double>> tripletList2;
    tripletList2.reserve(v.size());
    
    for(i=0;i<v.size();i++){
         // Controllo per l'indice i-3 (in direzione su)
        if (i - 3 >= 0) {
            tripletList2.push_back(Eigen::Triplet<double>(i, i - 3, -3.0));
        }
        // Controllo per l'indice i-1 (in direzione sinistra)
        if (i - 1 >= 0) {
            tripletList2.push_back(Eigen::Triplet<double>(i, i - 1, -1.0));
        }
        // Indice i
        tripletList2.push_back(Eigen::Triplet<double>(i, i, 9.0));
        // Controllo per l'indice i+1 (in direzione destra)
        if (i + 1 < v.size()) {
            tripletList2.push_back(Eigen::Triplet<double>(i, i + 1, -3.0));
        }
        // Controllo per l'indice i+3 (in direzione giù)
        if (i + 3 < v.size()) {
            tripletList2.push_back(Eigen::Triplet<double>(i, i + 3, -1.0));
        }
    }
    A2.setFromTriplets(tripletList2.begin(), tripletList2.end());
    //std::cout<<A2.topLeftCorner(10,10)<<std::endl;
    if (A2.isApprox(A2.transpose())) {
        std::cout << "La matrice A2 è simmetrica." << std::endl;
    } else {
        std::cout << "La matrice A2 non è simmetrica." << std::endl;
    }

    Eigen::VectorXd t6 = A2*w;
    MatrixXd t6matrix(height,width);  

    for(i=0;i<height;i++){
        for(j=0;j<width;j++){
            double value = t6(i * width + j);
            value = std::max(0.0, std::min(255.0, value));  // Clamp between 0 and 255
            t6matrix(i, j) = value;
        }
    }
    
    Matrix<unsigned char, Dynamic, Dynamic, Eigen::RowMajor> t6image(height, width);

    t6image = t6matrix.unaryExpr([](double val) -> unsigned char{
        return static_cast<unsigned char> (val);
    });

    const std::string output_image_patht6 = "t6image.png";
    // Save the noisy image
    if (stbi_write_png(output_image_patht6.c_str(), width, height, 1, t6image.data(), width) == 0) {
        std::cerr << "Error: Could not save output image" << std::endl;
        return 1;
    }


    




    //we already have the exact solution wich is v
    
    /*
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>solver(A1);
    solver.compute(A1);
    if(solver.info()!=Eigen::Success){
        cout << "cannot factorize the matrix" << endl;
        return 0;
    }
    
    */
    
    return 0;
}
