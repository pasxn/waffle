//http://man7.org/linux/man-pages/man3/dlopen.3.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

float* add_run(float num) {
  void *handle;
  float* (*plus_one)(float);
  char *error;

  handle = dlopen("./lib.so", RTLD_LAZY);

  dlerror();

  plus_one = (float* (*)(float)) dlsym(handle, "plus_one");

  float* result = (*plus_one)(num);
  dlclose(handle);

  printf("%f\n", *result);

  return result;
}

extern "C" {
  void run(float num) {
    add_run(num);
  }
}

int main() {
  float* z = (float*)malloc(sizeof(float));
  
  z = add_run(1.0f);

  printf("%f\n", *z);

  free(z);
}
