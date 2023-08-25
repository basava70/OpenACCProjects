#include <chrono>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  const long int N = 1000000; // 1e6
  float a = 3.0f;
  float x[N], y[N];

  // Initatizing the data
  for (int i = 0; i < N; ++i) {
    x[i] = 2.0f;
    y[i] = 1.0f;
  }
  
  // openacc kernel
  auto begin = std::chrono::high_resolution_clock::now();
#pragma acc kernels
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time elapsed for acc kernels: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                   .count()
            << std::endl;

  // checking for correction
  double tolerance = 1e-14;
  for (size_t i = 0; i < N; i++) {
    if (fabs((y[i] - 7.0)) > tolerance) {
      std::cout << " Data didn't match y[" << i << "] = " << y[i]
                << " and not 7" << std::endl;
    }
  }

  // basic cpu code
  auto begin_cpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::cout << "Time elapsed for cpu: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu -
                                                                    begin_cpu)
                   .count()
            << std::endl;
  std::cout << " ------------------------ " << std::endl;
  std::cout << " ------- Success -------- " << std::endl;
  std::cout << " ------------------------ " << std::endl;
}
