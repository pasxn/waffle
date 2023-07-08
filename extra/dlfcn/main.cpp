// http://man7.org/linux/man-pages/man3/dlopen.3.html

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <gnu/lib-names.h> 

int main(void) {
  void *handle;
  float* (*plus_one)(float);
  char *error;

  handle = dlopen("./lib.so", RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "%s\n", dlerror());
    exit(EXIT_FAILURE);
  }

  dlerror();

  plus_one = (float* (*)(float)) dlsym(handle, "plus_one");

  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "%s\n", error);
    exit(EXIT_FAILURE);
  }

  printf("%f\n", *(*plus_one)(2.3));
  dlclose(handle);
  exit(EXIT_SUCCESS);
}
