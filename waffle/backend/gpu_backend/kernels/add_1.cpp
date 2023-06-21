#include <stdio.h>
#include <chrono>
#include "V3DLib.h"
#include <CmdParameters.h>
#include "Support/Settings.h"

using namespace V3DLib;

V3DLib::Settings settings;

void add(Int n, Int::Ptr x, Int::Ptr y, Int::Ptr z) {
  For (Int i = 0, i<n, i+=16)
    Int a = x[i];
    Int b = y[i];
    z[i] = a + b;
  End
}

void addArrays(int size, const int* array1, const int* array2, int* result) {
    for (int i = 0; i < size; i++) {
    	int ca = array1[i];
    	int cb = array2[i];
    	result[i] = ca + cb;
    }
}

int main(int argc, const char *argv[]) {
  
  int size = 50;
//GPU CODE

  Int::Array a(size);
  Int::Array b(size);
  Int::Array r(size);        // Allocate and initialise the arrays shared between ARM and GPU
  srand(0);
  for (int i = 0; i < size; i++) {
    a[i] = 1;
    b[i] = 1;
  }
  
  auto start = std::chrono::high_resolution_clock::now();  //time starts here
 
  settings.init(argc, argv);

  auto k = compile(add);                 // Construct the kernel

  k.load(size, &a, &b, &r);                    // Invoke the kernel
  settings.process(k);
  
  auto end = std::chrono::high_resolution_clock::now(); //time ends here 
  
  std::chrono::duration<double> duration = end - start;
  
/*  printf(".......... GPU output ..........\n");
  for (int i = 0; i < size; i++){           // Display the result
    printf("GPU: add(%i, %i) = %i\n", a[i], b[i], r[i]);
  } */
  
  //from this point cpu code
  
  int array1[size];
  int array2[size];
  int result[size];
  for(int i=0; i<size; i++){
    array1[i]=1;
    array2[i]=1;
  }
  auto start_cpu = std::chrono::high_resolution_clock::now();  //time start
  addArrays(size, array1, array2,result);
  auto end_cpu = std::chrono::high_resolution_clock::now(); //time end 
  
  std::chrono::duration<double> duration_cpu = end - start;
  
/*  printf(".......... CPU output ..........\n");  
  for (int i = 0; i < size; i++) {
   printf("CPU: add(%i, %i) = %i\n", array1[i], array2[i], result[i]);
  } */
  printf(".........Execution Time.........\n");
  printf("Execution time for CPU: %f seconds\n", duration_cpu.count());
  printf("Execution time for GPU: %f seconds\n", duration.count());
  
  
  return 0;
}
