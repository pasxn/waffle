#include <stdio.h>
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

int main(int argc, const char *argv[]) {
  settings.init(argc, argv);

  auto k = compile(add);                 // Construct the kernel
  
  int size = 512;

  Int::Array a(size);
  Int::Array b(size);
  Int::Array r(size);        // Allocate and initialise the arrays shared between ARM and GPU
  srand(0);
  for (int i = 0; i < size; i++) {
    a[i] = 100 + (rand() % 100);
    b[i] = 100 + (rand() % 100);
  }

  k.load(size, &a, &b, &r);                    // Invoke the kernel
  settings.process(k);

  for (int i = 0; i < size; i++)           // Display the result
    printf("add(%i, %i) = %i\n", a[i], b[i], r[i]);
  
  return 0;
}
