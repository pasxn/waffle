//http://man7.org/linux/man-pages/man3/dlopen.3.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

void add_run(float num, float* result) {
  void *handle;
  float* (*plus_one)(float);
  char *error;

  handle = dlopen("./lib.so", RTLD_LAZY);

  dlerror();

  plus_one = (float* (*)(float)) dlsym(handle, "plus_one");

  result = (*plus_one)(num);
  dlclose(handle);

  printf("result: %f\n", *result);
}

extern "C" {
  void run(float num, float* result) {
    add_run(num, result);
  }
}

int main() {
  float* z = (float*)malloc(sizeof(float));
  
  add_run(1.0f, z);

  printf("%f\n", *z);

  free(z);
}
