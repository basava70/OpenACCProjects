/**
 * This code demonstrates simple reduction code
 * done in OpenACC without using any reduce directives.
 * Created on August 23rd, 2023.
 */

#include <iostream>
#include <string>

int main() {
  // delcaring and defining the variable
  int a = 0;
  // OpenACC kernel directive
#pragma acc kernels
  for (int i = 0; i < 5; i++)
    a++;

  // printing the output
  std::cout << " a : " << a << std::endl;

  return 0;
}
