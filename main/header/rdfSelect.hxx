#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include "types.hxx"



__global__ void unarySelect (circularBuffer<int> src, int* value, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size, int storeSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;
	
	if (src.pointer[newIndex] == (*value)) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}

__global__ void binarySelect (circularBuffer<int> src1, circularBuffer<int> src2, int* value1, int* value2, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size, int storeSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src1.end - src1.begin + src1.size) % src1.size) ) {
		return;
	}		

	int newIndex = (src1.begin + index) % src1.size;
	if ((src1.pointer[newIndex] == (*value1)) && (src2.pointer[newIndex] == (*value2))) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}



/*
* Make multiple select query, with specified comparison condition,
* on a triple store. Both queries and the store are supposed to 
* be already on the device. 
* 
* @param d_selectQueries : the array in which are saved the select values
* @param d_storePointer : pointer on the device to the triple store
* @param storeSize : size of the triple store
* @param comparatorMask : array of triple of comparator that are applied to the queries
*			must be of the size of the d_selectQueries
* @return a vector of type mem_t in which are saved the query results.
*/
std::vector<mem_t<tripleContainer>*> rdfSelect(const std::vector<tripleContainer*> d_selectQueries, 
		deviceCircularBuffer d_pointer,
		const int storeSize, 
		std::vector<int*> comparatorMask,
		std::vector<int>  arrs) 
{
	standard_context_t context;
	//Initialize elements
	int querySize =  d_selectQueries.size();
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	int* currentSize;
	cudaMalloc(&currentSize, sizeof(int));
	int* zero = (int*) malloc(sizeof(int));
	*zero = 0;

	int* finalResultSize  = (int*) malloc(sizeof(int));

	//Cycling on all the queries
	for (int i = 0; i < querySize; i++) {
		//Save variable to pass to the lambda operator
		tripleContainer* currentPointer = d_selectQueries[i];
		
		mem_t<tripleContainer>* currentResult = new mem_t<tripleContainer>(storeSize, context);

		int gridSize = 300;
	        int blockSize = (storeSize / gridSize) + 1;
		cudaMemcpy(currentSize, zero, sizeof(int), cudaMemcpyHostToDevice);
			
		switch(arrs[i]) {

			case(0): {
				int* value = &(currentPointer->subject);

				unarySelect<<<gridSize,blockSize>>>(d_pointer.subject, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(1): {
				int* value = &(currentPointer->predicate);
			
				unarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}
						
			case(2): {
                                int* value = &(currentPointer->object);

                                unarySelect<<<gridSize,blockSize>>>(d_pointer.object, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);
                                break;
			}
			
			case(3): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->predicate);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.predicate, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(4): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(5): {
				int* value1 = &(currentPointer->predicate);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}
						
			case(6): {
				cudaMemcpy(currentResult->data(), d_pointer.rdfStore.pointer, storeSize * sizeof(tripleContainer), cudaMemcpyDeviceToDevice);
				cudaMemcpy(currentSize, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
                                break;
			}
			
			
			default: {
				printf("ERROR ERRROR ERROR ERROR ERROR ERROR ERROR");
			}


		}
				
                cudaMemcpy(finalResultSize, currentSize, sizeof(int), cudaMemcpyDeviceToHost);
		currentResult->setSize(*finalResultSize);
		finalResults.push_back(currentResult);
	}
	cudaFree(currentSize);
	
	return finalResults;
}



