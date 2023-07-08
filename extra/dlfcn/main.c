#include <stdio.h>
#include <dlfcn.h>

typedef void (*addptr)(float, float, float*);

int main() {
    void* libraryhandle;
    addptr myadd;

    libraryhandle = dlopen("/lib.so", RTLD_LAZY);
    myadd = (addptr)dlsym(libraryhandle, "add");
    
    float* z;

    myadd(1.0, 1.0, z);
    printf("Result: %f\n", *z);

    dlclose(libraryhandle);
}
