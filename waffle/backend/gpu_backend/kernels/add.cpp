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

auto kernel = compile(add);

void run_kernel(int size, float* x, float* y, float* z) {
  Float::Array a(size);
  Float::Array b(size);
  Float::Array r(size);

  for(int i = 0; i < size; i++) {
    a[i] = x[i]; b[i] = y[i];
  }

  kernel.setNumQPUs(settings.num_qpus);
  kernel.load(size, &a, &b, &r);
  settings.process(kernel);

  for(int i = 0; i < size; i++) z[i] = r[i];
}

int main(int argc, const char *argv[]) {}

extern "C" {
  void run(int size, float* x, float* y, float* z) {
    run_kernel(size, x, y, z);
  }
}
