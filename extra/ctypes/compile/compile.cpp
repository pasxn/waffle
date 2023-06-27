// C code
extern "C" {
  void compile_kernel();
  int run_kernel(int val);

  void compile() {
    compile_kernel();
  }

  void run(int val, int* ret) {
    *ret = run_kernel(val);
  }
  
}


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

foo* kernel = new foo(5);

void compile_kernel() {
  kernel->compilee();
}

int run_kernel(int val) {
  return kernel->run(val);
}
