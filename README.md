# OpenACC Projects

Following are a collection of my OpenACC projects.

### 1. Hello

- A basic code to see how acc kernels work

### 2. Saxpy

- Saxpy code which compares the cpu version vs the OpenACC version
- Computation time
  | Version | Time in nanoseconds |
  | ------- | ------------------- |
  | CPU | 978989 |
  | OpenACC | 92279482 |
- The reason for the excessive computation time is the number of data copied in the kernels
  is much higher then the overhead of the calculation.
### 3. Jacobi Iteration
- Jacobi Iteration, is an updated iteration that takes the average of the four neighbouring elemens of a given element.
 For example, B[i][j] = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1])/4
  We consider the arrays A and B of size 1000 * 2000 with a given initial conditions. We use an error tolerance of 0.02.
We consider three kernels:
  - CPU kernel
  - OpenACC basic kernel *using just* ```#pragma acc kernels```
  - OpenACC advanced kernel *using* ```#pragma data copy(A), create(B)``` and ```#pragma reduction(max:error)```
- Computation time
-
  | Version | Time in seconds |
  | ------- | ------------------- |
  | CPU | 18 |
  | OpenACC basic | 33 |
  | OpenACC advanced | 0.87 |
- As we can the computation time in the basic openacc kernels almost doubled due to the constant copy in and copy out of the data arrays
  between host and device every iteration, which is highly redundant.
- In the final OpenACC advaned not only, we remove this redundancy by using ```#pragma data copy(A), create(B)``` which limits the data transfer
  to one time for the entire while loop.
   - Secondly, we also use ```#pragma acc kernels loop reduction(max:error)``` to use the existing reduction to get the max error
     parallelly.
 
