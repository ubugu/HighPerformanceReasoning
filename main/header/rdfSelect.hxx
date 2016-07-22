#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include "types.hxx"


//Kernel for select with one element to compare
__global__ void unarySelect (bufferPointer<int> src, int* value, tripleContainer* dest, bufferPointer<tripleContainer> store, int* size, int storeSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin + BUF_LEN) % BUF_LEN) ) {
		return;
	}	

	int newIndex = (src.begin + index) % BUF_LEN;
	
	if (src.pointer[newIndex] == (*value)) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}

//Kernel for select with two element to compare
__global__ void binarySelect (bufferPointer<int> src1, bufferPointer<int> src2, int* value1, int* value2, tripleContainer* dest, bufferPointer<tripleContainer> store, int* size, int storeSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src1.end - src1.begin + BUF_LEN) % BUF_LEN) ) {
		return;
	}		

	int newIndex = (src1.begin + index) % BUF_LEN;
	if ((src1.pointer[newIndex] == (*value1)) && (src2.pointer[newIndex] == (*value2))) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}



