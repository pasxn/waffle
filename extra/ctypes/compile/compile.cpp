// C++ code
#include <stdio.h>

class foo {
public:
  int data;

  foo(int value) {
    data = value;
  }

  void compilee() {
    data = data + 10;
  }

  int run(int value) {
    return data + value;
  }
};

foo* kernel = NULL;

void compile_kernel() {
  kernel = new foo(5);
  kernel->compilee();
}

int run_kernel(int val) {
  return kernel->run(val);
}

// C code
extern "C" {
  void compile() {
    compile_kernel();
  }

  void run(int val, int* ret) {
    *ret = run_kernel(val);
  }
  
}
