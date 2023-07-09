#include <stdio.h>
#include <chrono>
#include "V3DLib.h"
#include <CmdParameters.h>
#include "Support/Settings.h"

using namespace V3DLib;

V3DLib::Settings settings;

void add(Int rows, Int columns, Int::Ptr x, Int::Ptr y, Int::Ptr z) {
  Int n= rows*columns;
  For (Int i=0, i<n, i+=16)
    Int a = *x;
    Int b = *y;
    *z = a+b;
    x = x+16;
    y = y+16;
    z = z+16;
  End
}


int main(int argc, const char *argv[]) {

	int columns =5;
	int rows = 5;
	int size = rows*columns;
	int iterations = 5;
	
	Int::Array a(rows*columns);
  Int::Array b(rows*columns);
  Int::Array r(rows*columns);

  for (int i = 0; i < (rows*columns); i++) {
    a[i] = i;
    b[i] = i;
  }
  auto k = compile(add);

  for(int y = 0; y < iterations ; y++) {
    k.load(rows, columns, &a, &b, &r);
  }
  settings.process(k);
  
  for(int i=0;i<(rows*columns);i++){
  	printf("The value of results is %i\n",r[i]);
  
  }
  
  return 0;
  
}

/*
//kernel
void add(Int::Ptr x, Int::Ptr y, Int::Ptr z , Int rows, Int cols) {
  For (Int i = 0, i<(rows*cols), i+=16)
    //For(Int j=0, j<cols, i+=16)
      Int a = *x;
      Int b = *y;
      *z = a + b;
   // End
  End
}

//cpp function
void addArrays(int *arr1, int *arr2, int *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            *(result + i * cols + j) = *(arr1 + i * cols + j) + *(arr2 + i * cols + j);
        }
    }
}


int main(int argc, const char *argv[]) {
	int rows= 17;
	int cols= 17;
	
	int track=0;
	
	int tot = rows*cols;
	
	Int::Array a(tot);
	Int::Array b(tot);
	Int::Array r(tot);
	
	
	for (int i = 0; i < (tot); i++) {
    	    a[i] = 30;
    	    b[i] = 40;
	}
	
	auto k = compile(add);

  	for(int y = 0; y < (tot) ; y++) {
	    k.load(&a,&b,&r,rows,cols);
	}
	settings.process(k);
	
	printf("GPU Execution\n");
	for(int i=0;i<tot;i++){
        	printf("%i ",r[i]); 
  	}
//	for (int i = 0; i < rows; i++) {
  //          for (int j = 0; j < cols; j++) {
    //            printf("%i ",r[track]);
      //          track++;
        //    }
          //  printf("\n");
//    	}
    	
    	
    	int arr1[rows][cols] = {{0}};
   	int arr2[rows][cols] = {{0}};
	int result[rows][cols] = {{0}};
    
	for(int i =0; i<rows; i++){
	    for(int j=0; j<cols; j++){
    		arr1[i][j] = 2;
    		arr2[i][j] = 3;    	
    	    }
    	}
    	
    	addArrays((int *)arr1, (int *)arr2, (int *)result, rows, cols);
    	printf("CPU execution\n");
    	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }
  
	return 0;
	
  
}

*/

