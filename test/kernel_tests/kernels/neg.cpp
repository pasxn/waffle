#include <stdio.h>
#include <chrono>
#include "V3DLib.h"
#include <CmdParameters.h>
#include "Support/Settings.h"

using namespace V3DLib;

V3DLib::Settings settings;

//kernel
void neg(Int n, Int::Ptr x , Int::Ptr y) {
  For (Int i = 0, i<n, i+=16)
    Int a = x[i];
    y[i] = 0-a;
  End
}

//cpp function
void negArrays(int size, const int* aa, int* rr) {
    for (int i = 0; i < size; i++) {
    	aa[i] = 0-rr[i];
    }
}

int main(int argc, const char *argv[]) {
  int size = 900000;
  int iterations = 1;
//GPU arrays
  Int::Array a(size);
  Int::Array r(size);
  for (int i = 0; i < size; i++) {
    a[i] = 5;
  }
 
 //CPU arrays 
  int aa[size];
  int rr[size];
  for(int i=0; i<size; i++){
    aa[i]=5;
  }
  
  
  //GPU execution
  settings.init(argc, argv);

  auto k = compile(neg);
  k.setNumQPUs(settings.num_qpus);

  auto start = std::chrono::high_resolution_clock::now();  
  for(int y=0; y<iterations ;y++){            
    k.load(size, &a,&r);  
  }
  settings.process(k);
  auto end = std::chrono::high_resolution_clock::now(); 
  
  std::chrono::duration<double> duration = end - start;


//CPU execution
  auto start_cpu = std::chrono::high_resolution_clock::now();  //time start
  for(int y=0; y<iterations; y++){
    negArrays(size, aa,rr);
  }
  auto end_cpu = std::chrono::high_resolution_clock::now(); //time end 
  
  std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
  
  // functional verification
  for(int j = 0; j < size; j++) {
    if(r[j] != rr[j])
      printf("CPU output and GPU output is not equal at j = %d \n", j);
  }  
  
  printf(".........Execution Time.........\n");
  printf("Execution time for CPU: %f seconds\n", duration_cpu.count());
  printf("Execution time for GPU: %f seconds\n", duration.count());
  
  
  return 0;
}
