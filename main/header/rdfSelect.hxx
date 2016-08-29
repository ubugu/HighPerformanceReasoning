#pragma once

#include <cstdlib>

#include "types.hxx"
#include "operations.hxx"


enum class SelectArr { S = 1, P = 2, O = 4, SP = 3, SO = 5, PO = 6, SPO = 7};

__global__ void unarySelect (CircularBuffer<TripleContainer> src, int target_pos, int first_pos, int second_pos, size_t value, size_t* dest, int width, int* size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;

	size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};

	if (temp[target_pos] == value) {
		int add = atomicAdd(size, 1);
		size_t* dest_p = (size_t*) (dest + add * width) ;
		*dest_p = temp[first_pos];
		*(dest_p + 1) = temp[second_pos];

	}
}

__global__ void binarySelect (CircularBuffer<TripleContainer> src, int target_pos, int target_pos2, int dest_pos, size_t value, size_t value2, size_t* dest, int width, int* size) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
			return;
		}	

		int newIndex = (src.begin + index) % src.size;

		size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};

		if ((temp[target_pos] == value) && (temp[target_pos2] == value2)) {
			int add = atomicAdd(size, 1);
			size_t* dest_p = (size_t*) (dest + add * width) ;
			*dest_p = temp[dest_pos];	
		}
}


class SelectOperation : public Operation
{
	private:
		std::vector<size_t> constants;
		int arr;
		
	public:
		SelectOperation(std::vector<size_t> constants, std::vector<std::string> variables, int arr) {
			this->variables = variables;
			this->constants = constants;	
			this->arr = arr;
		}

		int getArr() {
			return this-> arr;
		}
			
		std::vector<size_t> getQuery() {
			return this->constants;
		}

		void rdfSelect(CircularBuffer<TripleContainer> d_pointer, const int storeSize) {
			
			//Initialize elements	
			int* d_resultSize;
			int h_resultSize  = 0;
			cudaMalloc(&d_resultSize, sizeof(int));
			cudaMemcpy(d_resultSize, &h_resultSize, sizeof(int), cudaMemcpyHostToDevice);
	
			//INSERIRE DIVISIONE CORRETTA
			int gridSize = 300;
			int blockSize = (storeSize / gridSize) + 1;
		
			result = new Binding(variables.size(), storeSize);
						
			switch(arr) {

				case(1): {
					unarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 1, 2, constants[0], result->pointer, result->width, d_resultSize);
					break;
				}

				case(2): {
					unarySelect<<<gridSize,blockSize>>>(d_pointer,  1, 0, 2, constants[0], result->pointer, result->width, d_resultSize);
					break;
				}
					
				case(4): {
			        unarySelect<<<gridSize,blockSize>>>(d_pointer,  2, 0, 1, constants[0], result->pointer, result->width, d_resultSize);
			        break;
				}
		
				case(3): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 1, 2, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
				case(5): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 2, 1, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
				case(6): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 1, 2, 0, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
					/*
				case(7): {
					cudaMemcpy(result->data(), d_pointer.rdfStore.pointer, storeSize * sizeof(TripleContainer), cudaMemcpyDeviceToDevice);
					cudaMemcpy(d_resultSize, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
			                break;
				}*/
		
			}
	
	
			cudaMemcpy(&h_resultSize, d_resultSize, sizeof(int), cudaMemcpyDeviceToHost);
			
			result->height  =  h_resultSize;
					
			cudaFree(d_resultSize);
		}
	
};


