#include <stdio.h>
#include "V3DLib.h"
#include <CmdParameters.h>
#include "Support/Settings.h"

using namespace V3DLib;

V3DLib::Settings settings;

void add(Int n, Int::Ptr x, Int::Ptr y, Int::Ptr z) {
  For (Int i = 0, i < n, i += 16)
    Int a = x[i];
    Int b = y[i];
    z[i] = a + b;
  End
}

auto kernel = NULL;

void compile_kernel() {
  kernel = compile(add);
}

void run_kernel(int size) {
  Int::Array a(size);
  Int::Array b(size);
  Int::Array r(size);

  kernel.setNumQPUs(settings.num_qpus);
  kernel.load(size, &a, &b, &r);
  settings.process(kernel);
}

extern "C" {
  void _compile() {
    compile_kernel();
  }

  void run(int size, int* ret) {
    run_kernel(val);
  }
  
}
