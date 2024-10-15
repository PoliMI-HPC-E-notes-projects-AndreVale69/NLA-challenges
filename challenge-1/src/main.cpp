#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Eigen/Sparse>
// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;
using namespace std;

// Funzione per eseguire il zero padding su image_data
std::vector<unsigned char> zero_pad_image(const unsigned char* image_data, int width, int height) {
    int new_width = width + 2;   // Larghezza dopo padding
    int new_height = height + 2; // Altezza dopo padding

    // Creiamo un nuovo array per l'immagine con padding
    std::vector<unsigned char> padded_image(new_width * new_height, 0); // Inizializzato con zeri

    // Copiamo i dati dell'immagine originale dentro l'immagine con padding
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Calcoliamo l'indice nell'immagine originale
            int original_idx = i * width + j;
            // Calcoliamo l'indice corrispondente nell'immagine con padding (shiftato di 1 riga e 1 colonna)
            int padded_idx = (i + 1) * new_width + (j + 1);
            // Copiamo il valore del pixel
            padded_image[padded_idx] = image_data[original_idx];
        }
    }

    return padded_image;
}


int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  int max = 50;
  int min = -50;

  const char* input_image_path = argv[1];
  const char* input_image_path_noise = argv[2];

  // Load the image using stb_image
  int width, height, channels;
  int width_noise, height_noise, channels_noise;
  // for greyscale images force to load only one channel
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  unsigned char* image_data_noise = stbi_load(input_image_path_noise, &width_noise, &height_noise, &channels_noise, 1);
  if (!image_data || !image_data_noise) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

   // Fill the matrices with image data
  // for (int i = 0; i < height*width; ++i) {

  //   int tmp = image_data[i] + (rand() % (max-min+1) + min);

  //   if(tmp>255){
  //     tmp = 255;
  //   }else if(tmp < 0){
  //     tmp = 0;
  //   }
  //   image_data[i] = tmp;
  // }

   // Numero totale di pixel
    int num_pixels = width * height;

    // Creiamo un vettore Eigen di dimensioni num_pixels e copiamo i dati dell'immagine
    Eigen::VectorXd v(num_pixels);
    for (int i = 0; i < num_pixels; ++i) {
        v(i) = static_cast<double>(image_data[i]); // Converti da unsigned char a double
    }
    Eigen::VectorXd w(num_pixels);
    for (int i = 0; i < num_pixels; ++i) {
        w(i) = static_cast<double>(image_data_noise[i]); // Converti da unsigned char a double
    }
  
  if(v.size() == height*width){
    std::cout<<"are same!!\n";
    cout<<"v.size(): "<<v.size()<<" = "<<"height*width: "<<height*width<<endl;
  }
  cout<<"norm of the vetor: "<<v.norm()<<endl;

  std::cout << "Image loaded: " << height << "x" << width << " with " << channels << " channels." << std::endl;



// Applichiamo zero padding
    std::vector<unsigned char> padded_image = zero_pad_image(image_data, width, height);


 int size = height * width;
    
    // Creiamo una matrice sparsa A_H
    SparseMatrix<double, RowMajor> A_H(size, size);
    std::vector<Triplet<double>> tripletList;
    tripletList.reserve(size);
    // Filtro 3x3 (media mobile)
    double weight = 1.0 / 9.0;
    // Applichiamo il filtro 3x3 su ogni elemento della matrice
    for (int i = 0; i < height ; i++) {
        for (int j = 0; j < width; j++) {
            int pos = i * width + j;  // Posizione dell'elemento nella matrice sparsa

            // Aggiungiamo il valore centrale (m_ij)
            tripletList.push_back(Triplet<double>(pos, pos, 1.0/9));

            // Aggiungiamo i vicini (sopra, sotto, sinistra, destra, e diagonali)
            if (i > 0) {
            tripletList.push_back(Triplet<double>(pos, (i - 1) * width + j, weight));     // Sopra
            }
            if(i<height-1){
            tripletList.push_back(Triplet<double>(pos, (i + 1) * width + j, weight));     // Sotto
            }
            if(j>0){
            tripletList.push_back(Triplet<double>(pos, i * width + (j - 1), weight));     // Sinistra
            }
            if(j<width-1){
            tripletList.push_back(Triplet<double>(pos, i * width + (j + 1), weight));     // Destra
            }
            // Diagonali
            if(i >0 && j > 0){
            tripletList.push_back(Triplet<double>(pos, (i - 1) * width + (j - 1), weight)); // Sopra-Sinistra
            }
            if(i>0 && j<width-1){
            tripletList.push_back(Triplet<double>(pos, (i - 1) * width + (j + 1), weight)); // Sopra-Destra
            }
            if(i<height-1 && j>0){
            tripletList.push_back(Triplet<double>(pos, (i + 1) * width + (j - 1), weight)); // Sotto-Sinistra
            }
            if(i<height-1 && j<width-1){
            tripletList.push_back(Triplet<double>(pos, (i + 1) * width + (j + 1), weight)); // Sotto-Destra
            }        
        }
    }

cout<<"fin qui vengo!!\n";
  

  A_H.setFromTriplets(tripletList.begin(), tripletList.end());
   // Stampiamo la matrice sparsa A_H
    std::cout << "Matrice sparsa A_H non zero: \n" << A_H.nonZeros() << std::endl;

// Eigen::VectorXd t4 = A_H*v;//task 4


// task 5 :
//   Eigen::VectorXd smoothed_img = A_H*w;
// vector<unsigned char> out_smoothed(size);

//   for(int i = 0; i<size; i++){
//     out_smoothed[i] = static_cast<unsigned char>(smoothed_img[i]);
    

//   }

  
//   if(stbi_write_png("output_image_smoothed.png", width_noise, height_noise, channels_noise, out_smoothed.data() , width_noise * channels_noise) == 0){
//     std::cerr << "Error: Could not save  image" << std::endl;

//     return 1;

//   }

  // task 6:

   // Creiamo una matrice sparsa A_H
    SparseMatrix<double, RowMajor> A2(size, size);
    std::vector<Triplet<double>> triplet_A2;
    triplet_A2.reserve(size);

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int pos = i * width + j;  // Posizione dell'elemento nella matrice sparsa

            // Aggiungiamo il valore centrale (m_ij)
            triplet_A2.push_back(Triplet<double>(pos, pos, 9));

            // Aggiungiamo i vicini (sopra, sotto, sinistra, destra, e diagonali)
            triplet_A2.push_back(Triplet<double>(pos, (i - 1) * width + j, -3));     // Sopra
            triplet_A2.push_back(Triplet<double>(pos, (i + 1) * width + j, -1));     // Sotto
            triplet_A2.push_back(Triplet<double>(pos, i * width + (j - 1), -1));     // Sinistra
            triplet_A2.push_back(Triplet<double>(pos, i * width + (j + 1), -3));     // Destra

           
        }
    }

  A2.setFromTriplets(triplet_A2.begin(), triplet_A2.end());
   // Stampiamo la matrice sparsa A_H
    std::cout << "Matrice sparsa A2 non zero: \n" << A2.nonZeros() << std::endl;




  // if(stbi_write_png("output_image.png", width, height, channels, image_data, width * channels) == 0){
  //   std::cerr << "Error: Could not save  image" << std::endl;

  //   return 1;

  // }

    // Libera la memoria
    stbi_image_free(image_data);
    stbi_image_free(image_data_noise);
    //try push request
    return 0;
}