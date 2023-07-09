extern "C" {
  void add_kernel(int size, float* x, float* y, float* z);

  void add(int size, float* x, float* y, float* z) {
    add_kernel(size, x, y, z);
  }
  
}

void add_kernel(int size, float* x, float* y, float* z) {
  for (int i = 0; i < size; i++)
    z[i] = x[i] + y[i];
}