#include <stdio.h>
#include <chrono>
#include "V3DLib.h"
#include <CmdParameters.h>
#include "Support/Settings.h"

using namespace V3DLib;

V3DLib::Settings settings;

void neg(Int n, Int::Ptr x , Int::Ptr y) {
  For (Int i = 0, i<n, i+=16)
    Int a = x[i];
    y[i] = 0-a;
  End
}

void negArrays(int size, const int* array1, int* result) {
    for (int i = 0; i < size; i++) {
    	result[i] = -array1[i];
    }
}

int main(int argc, const char *argv[]) {
  
  int size = 50;
//GPU CODE

  Int::Array a(size);
  Int::Array b(size);
  
  for (int i = 0; i < size; i++) {
    a[i] = 5;
  }
  
  auto start = std::chrono::high_resolution_clock::now();  //time starts here
 
  settings.init(argc, argv);

  auto k = compile(neg);                 // Construct the kernel

  k.load(size, &a , &b);                    // Invoke the kernel
  settings.process(k);
  
  auto end = std::chrono::high_resolution_clock::now(); //time ends here 
  
  std::chrono::duration<double> duration = end - start;
  
  printf(".......... GPU output ..........\n");
  for (int i = 0; i < size; i++){           // Display the result
    printf("GPU: Neg(%i) = %i\n", a[i], b[i]);
  } 
  
  //from this point cpu code
  
  int array1[size];
  int result[size];
  for(int i=0; i<size; i++){
    array1[i]=5;
  }
  auto start_cpu = std::chrono::high_resolution_clock::now();  //time start
  negArrays(size, array1,result);
  auto end_cpu = std::chrono::high_resolution_clock::now(); //time end 
  
  std::chrono::duration<double> duration_cpu = end - start;
  
  printf(".......... CPU output ..........\n");  
  for (int i = 0; i < size; i++) {
   printf("CPU: Neg(%i) = %i\n", array1[i],result[i]);
  } 
  printf(".........Execution Time.........\n");
  printf("Execution time for CPU: %f seconds\n", duration_cpu.count());
  printf("Execution time for GPU: %f seconds\n", duration.count());
  
  
  return 0;
}
