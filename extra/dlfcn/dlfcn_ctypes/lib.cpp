// need to be C in order to use dlopen()
extern "C"{
  float*  plus_one(float num){
    float* z;
    *z = num + 1.0f;
    return z;
  }
}
