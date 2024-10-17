//
// Created by javed-abdullah on 10/16/24.
//
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <Eigen/Sparse>
// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "./external_libs/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./external_libs/stb_image_write.h"

using namespace Eigen;
using namespace std;


typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];
    // Load the image using stb_image
    int width, height, channels;

    // for greyscale images force to load only one channel
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);


    /* ============== TASK 8============ */
    int imageSize = 200;
    MatrixXd chessboeard(200,200);
    int ctbox = 0;
    // bool white = false;
    // bool black = true;
    // 0.0; //nero
    // 255.0;//bianco
    for(int k=0;k<8;k++) {
        int i_pos = k*25;
        int j_pos = 0;
        bool start_with_black;
        if(k%2==0) {
            start_with_black = true;
        }else {
            start_with_black = false;
        }
        bool black, white;
        if(start_with_black) {
            black = true;
            white = false;
        }else {
            black = false;
            white = true;
        }

        for(int p=0;p<8;p++) {
            for(int i=i_pos;i<i_pos+25;i++) {
                for(int j=j_pos;j<j_pos+25;j++) {
                    if(white) {
                        chessboeard(i,j) = 255.0;
                    }
                }
            }
            if(black) {
                white = true;
                black = false;
            }else {
                black = true;
                white = false;
            }
            j_pos+=25;;
        }
    }

    cout<<"norma della matrice: "<<chessboeard.norm()<<endl;
    // Converti la matrice in un array di unsigned char per stbi_write_png
    // unsigned char* image = new unsigned char[imageSize * imageSize];
    // for (int i = 0; i < imageSize; ++i) {
    //     for (int j = 0; j < imageSize; ++j) {
    //         image[i * imageSize + j] = static_cast<unsigned char>(chessboeard(i, j));
    //     }
    // }
    //
    // if(stbi_write_png("chessboard.png", 200, 200, 1, image, 200 * 1) == 0){
    //     std::cerr << "Error: Could not save  image" << std::endl;
    //
    //     return 1;
    //
    // }
    /* ==============END  TASK 8============ */


    /*  task 9*/
    int max = 50;
    int min = -50;
   // Fill the matrice with noise
  for (int i = 0; i < height*width; ++i) {

      int tmp = image_data[i] + (rand() % (max-min+1) + min);

      if(tmp>255){
          tmp = 255;
      }else if(tmp < 0){
          tmp = 0;
      }
      image_data[i] = tmp;
  }

    if(stbi_write_png("chessboeard_with_noise.png", width, height, channels, image_data , width * channels) == 0){
      std::cerr << "Error: Could not save  image" << std::endl;
      return 1;
    }



    return 0;
}