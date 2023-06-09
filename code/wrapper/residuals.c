#include <stdlib.h>
#include <stdio.h>
#include "residuals.h"
#include <math.h>

//void fastResiduals(double *X, double Z[][3], int ROWS, int ZCOLS, double bandwidth, double *residuals){
void fastResiduals(struct mystruct_t *data, int ROWS, int ZCOLS, double bandwidth, double *residuals){
  double k;

  for(int i = 0; i < ROWS ; i++){

    double summ = 0;
    double weight = 0;
    for(int j = 0; j < ROWS ; j++){
      
      // L2 norm
      double d = 0.0;
      for (int n = 0; n < ZCOLS; n++) {
            double diff = data->Z[i][n] - data->Z[j][n]; 
        
            d += diff * diff;
            }
      d = sqrt(d);

      // Uniform Kernel
      if (abs(d) <= bandwidth) {
        k = 1 / (2 * bandwidth);
      }
      else {
        k = 0;
      }
        
        summ += k * data->X[j];
        weight += k;
    }
    residuals[i] = data->X[i] - summ / weight;
  }
}

// void main(){

//   int ROWS = 3;
//   int ZCOLS = 3;
//   double bandwidth = 5.3;
//   double X[] = {2.5, 4, 1.4};
//   double Zmat[][3] = {{5.2, 0.7, 3.9}, {3.5, 1, 7.2}, {5, 0.4, 2.1}};
  
//   double residuals[3];
  

//   fastResiduals(X, Zmat, ROWS, ZCOLS, bandwidth, residuals);

//   for (int n = 0; n < ROWS; n++) {
//     printf("%.2f", residuals[n]);

//   }

// }



