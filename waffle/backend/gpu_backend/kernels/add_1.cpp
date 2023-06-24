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
  
  int size = 400000;
  int iterations = 1000;

  // GPU arrays
  Int::Array a(size);
  Int::Array b(size);
  Int::Array r(size);

  for (int i = 0; i < size; i++) {
    a[i] = 1;
    b[i] = 1;
  }

  // CPU arrays
  int aa[size];
  int bb[size];
  int rr[size];

  for(int i = 0; i < size; i++) {
    aa[i]=1;
    bb[i]=1;
  }

  // GPU ececution
  settings.init(argc, argv);
  auto k = compile(add);

  std::chrono::duration<double> accumilated_duration_gpu = 0;

  for(int y = 0; y < iterations ; y++) {
    auto start_gpu = std::chrono::high_resolution_clock::now();
    k.load(size, &a, &b, &r);
    settings.process(k);
    auto end_gpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_gpu = end_gpu - start_gpu;
    accumilated_duration_gpu += duration_gpu;
  }

  // CPU ececution
  std::chrono::duration<double> accumilated_duration_cpu = 0;

  for(int y = 0; y < iterations ; y++) {
    auto start_cpu = std::chrono::high_resolution_clock::now();
    addArrays(size, aa, bb,rr);
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    accumilated_duration_cpu += duration_cpu;
  }

  // functional verification
  for(int j = 0; j < size; j++) {
    if(r[j] != rr[j])
      printf("CPU output and GPU output is not equal at j = %d", j)
  }  

  // time log
  printf(".........Execution Time.........\n");
  printf("Execution time for CPU: %f seconds\n", accumilated_duration_cpu.count());
  printf("Execution time for GPU: %f seconds\n", accumilated_duration_gpu.count());
}

