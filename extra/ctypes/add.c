void add(int size, float* x, float* y, float* z) {
  for (int i = 0; i < size; i++)
    z[i] = x[i] + y[i];
}
