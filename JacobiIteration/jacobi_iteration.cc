
/**
 * The standard jacobi iteration to compare openacc
 * with sequential code.
 * Written on August 23rd, 2023.
 **/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <string>

#define width 1000
#define height 2000

typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_object;
const double tolerance = 1e-2;
const size_t iteration_check = 500;

// base matrix
double A[width + 2][height + 2]; // matrix that updates using the base matrix
double B[width + 2][height + 2];

time_object begin, end;

/*****************************************/
/** Initializing the data **/
void initalize_data() {
  for (size_t iter_x = 0; iter_x < width + 2; iter_x++) {
    for (size_t iter_y = 0; iter_y < height + 2; iter_y++) {
      A[iter_x][iter_y] = 0.0;
      B[iter_x][iter_y] = 0.0;
    }
  }

  for (size_t iter_x = 0; iter_x < width + 2; iter_x++) {
    A[iter_x][0] = 0.0;
    A[iter_x][height + 1] = (100.0 / width) * iter_x;
  }

  for (size_t iter_y = 0; iter_y < height + 2; iter_y++) {
    A[0][iter_y] = 0.0;
    A[width + 1][iter_y] = (100.0 / height) * iter_y;
  }
}

/*****************************************/
/** Printing the iteration details for every iteration_check **/
void print_details(const size_t iteration, const double error) {
  std::cout << " --- Iteration : " << iteration << " Error : " << error
            << " ---- " << std::endl;
}

/*****************************************/
/** Printing the total time elapsed **/
void print_time_elapsed() {
  double divider = 1.0;
  auto elapsed_time =
      std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
  // std::string type = "seconds";
  if (elapsed_time == 0) {
    elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    divider = 1000000.0;
    // type = "microseconds";
  }

  std::cout << "Time elapsed = " << (double)elapsed_time / divider << " seconds"
            << std::endl;
}

/*****************************************/
/** Jacobi iteration sequential function **/
void jacobi_iteration_sequential() {
  size_t iteration = 0;
  double error = 10;
  while (error > tolerance) {
    // running an iteration of jacobi for the length - 2
    for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
      for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
        B[iter_x][iter_y] =
            0.25 * (A[iter_x - 1][iter_y] + A[iter_x + 1][iter_y] +
                    A[iter_x][iter_y - 1] + A[iter_x][iter_y + 1]);
      }
    }

    // error calculation
    error = 0;

    for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
      for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
        error = fmax(fabs(B[iter_x][iter_y] - A[iter_x][iter_y]), error);
        A[iter_x][iter_y] = B[iter_x][iter_y];
      }
    }
    if (iteration % iteration_check == 0)
      print_details(iteration, error);

    iteration++;
  }
  std::cout << "End of jacobi sequential function" << std::endl;
  print_details(iteration, error);
}

/*****************************************/
/** Jacobi iteration basic method **/
void jacobi_iteration_openacc_basic() {
  size_t iteration = 0;
  double error = 10;
  while (error > tolerance) {
    // adding basic openacc suggestive kernels
#pragma acc kernels
    for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
      for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
        B[iter_x][iter_y] =
            0.25 * (A[iter_x - 1][iter_y] + A[iter_x + 1][iter_y] +
                    A[iter_x][iter_y - 1] + A[iter_x][iter_y + 1]);
      }
    }

    // error calculation
    error = 0;
    // adding basic openacc suggestive kernels
#pragma acc kernels
    for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
      for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
        error = fmax(fabs(B[iter_x][iter_y] - A[iter_x][iter_y]), error);
        A[iter_x][iter_y] = B[iter_x][iter_y];
      }
    }
    if (iteration % iteration_check == 0)
      print_details(iteration, error);

    iteration++;
  }
  std::cout << "End of jacobi sequential function" << std::endl;
  print_details(iteration, error);
}

/*****************************************/
/** Jacobi iteration advanced method **/
void jacobi_iteration_openacc_advanced() {
  size_t iteration = 0;
  double error = 10;

  // copying the data A only once from host before the while loop and copy
  // back to host when exiting the while loop.
  // For B, we dont copy at all, as it is not needed.
  // Reduces the data copying from 3328 (total iteration count) times to 1
  // time.
#pragma acc data copy(A), create(B)
  {
    while (error > tolerance) {
      // running an iteration of jacobi for the length - 2
#pragma acc kernels
      for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
        for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
          B[iter_x][iter_y] =
              0.25 * (A[iter_x - 1][iter_y] + A[iter_x + 1][iter_y] +
                      A[iter_x][iter_y - 1] + A[iter_x][iter_y + 1]);
        }
      }

      // error calculation
      error = 0;

// collapsing both the forloops into one and using the
// parallel reduction method with "max" as the operator on the error
// variable.
#pragma acc kernels loop collapse(2) reduction(max : error)
      for (size_t iter_x = 1; iter_x < width + 1; iter_x++) {
        for (size_t iter_y = 1; iter_y < height + 1; iter_y++) {
          error = fmax(fabs(B[iter_x][iter_y] - A[iter_x][iter_y]), error);
          A[iter_x][iter_y] = B[iter_x][iter_y];
        }
      }
      if (iteration % iteration_check == 0)
        print_details(iteration, error);

      iteration++;
    }
    std::cout << "End of jacobi sequential function" << std::endl;
    print_details(iteration, error);
  }
}

int main() {

  // sequential code
  std::cout << " ----------------------------------- " << std::endl;
  std::cout << " Starting sequential jacobi iteration" << std::endl;
  std::cout << " ----------------------------------- " << std::endl;
  initalize_data();
  begin = std::chrono::high_resolution_clock::now();
  jacobi_iteration_sequential();
  end = std::chrono::high_resolution_clock::now();
  print_time_elapsed();
  std::cout << " ------ Sequential code success !! ------ " << std::endl;
  std::cout << " " << std::endl;
  std::cout << " " << std::endl;
  //
  // openacc basic code
  std::cout << " ----------------------------------- " << std::endl;
  std::cout << " Starting basic openacc jacobi iteration" << std::endl;
  std::cout << " ----------------------------------- " << std::endl;
  initalize_data();
  begin = std::chrono::high_resolution_clock::now();
  jacobi_iteration_openacc_basic();
  end = std::chrono::high_resolution_clock::now();
  print_time_elapsed();
  std::cout << " ------ OpenACC basic code success !! ------ " << std::endl;
  std::cout << " " << std::endl;
  std::cout << " " << std::endl;

  // openacc advanced code
  std::cout << " ----------------------------------- " << std::endl;
  std::cout << " Starting advanced openacc jacobi iteration " << std::endl;
  std::cout << " ----------------------------------- " << std::endl;
  initalize_data();
  begin = std::chrono::high_resolution_clock::now();
  jacobi_iteration_openacc_advanced();
  end = std::chrono::high_resolution_clock::now();
  print_time_elapsed();
  std::cout << " ------ OpenACC advanced code success !! ------ " << std::endl;

  return 0;
}
