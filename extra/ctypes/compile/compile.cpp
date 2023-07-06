// C++ code
#include <stdio.h>

class foo {
public:
  int data;

  foo(int value) {
    data = value;
  }

  void compilee() {
    data = data + 5;
  }

  void run(int size, float *input, float *output) {
    for(size_t i=0; i<size; i++)
      output[i] = input[i] + (float)(data);
  }
};

foo* kernel = NULL;

void compile_kernel() {
  kernel = new foo(5);
  kernel->compilee();
}

void run_kernel(int size, float *input, float *output) {
  kernel->run(size, input, output);
}

// C code
extern "C" {
  void compile() {
    compile_kernel();
  }

  void run(int size, float *input, float *output) {
    run_kernel(size, input, output);
  }
  
}
