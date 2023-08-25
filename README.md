# OpenACC Projects

Following are a collection of my OpenACC projects.

### 1. Hello

- A basic code to see how acc kernels work

### 2. Saxpy

- Saxpy code which compares the cpu version vs the OpenACC version
- Computation time
  | Version | Time in nanoseconds |
  | :----: | :----:|
  | CPU | 978989 |
  | OpenACC | 92279482 |
- The extra computation time is because the time taken to copy the arrays between host and device is much more than the simple
  saxpy method.
### 3. Jacobi Iteration
- In this method, we use the standard jacobi iteration which is
   - B(i) = 0.25 * (A(i-1)(j) + A(i+1)(j) + A(i)(j-1) + A(i)(j+1))
- In our data of size 1000 * 2000 matrix, it took 3428 iterations to converge with an error tolerance of 0.02
- We use three kernels
    - CPU kernel
    - OpenACC basic *using just* ```#pragma acc kernels```
    - OpenACC advanced *using* ```#pragma acc data copy(A), create(B)``` and ```#pragma acc kernels reduction(max:error)```
- Computation time
  | Version | Time in seconds |
  | :----: | :----:|
  | CPU | 18 |
  | OpenACC basic | 33 |
  | OpenACC advanced | 0.87 |
- The **OpenACC basic** code suffers from the same problem as the one in saxpy, where we copy the data every iteration between host and device
- We rectify this issue in **OpenACC advanced** version by using ```#pragma acc data copy(A), create(B)```. In addition to this, we also
  take advantage of the inbuilt ```reduction(max)``` to make sure we get the error parallely.
