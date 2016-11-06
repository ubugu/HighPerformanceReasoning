#pragma once

#include <cstdlib>
#include <chrono>

#include "types.hxx"
#include "operations.hxx"


enum class SelectArr { SPO = 0, S = 1, P = 2, O = 4, SP = 3, SO = 5, PO = 6, };

//Select kernel for one constant select
__global__ void unarySelect (CircularBuffer<size_t> src, int target_pos, int first_pos, int second_pos, size_t value, size_t* dest,  int* nextIndex) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;

	if (src.pointer[newIndex * 3 + target_pos] == value) {
		int add = atomicAdd(nextIndex, 1);
		size_t* dest_p = (size_t*) (dest + add * 2) ;
		*dest_p = src.pointer[newIndex * 3 + first_pos];
		*(dest_p + 1) = src.pointer[newIndex * 3 + second_pos];
	}
}

//Select kernel for two constant select
__global__ void binarySelect (CircularBuffer<size_t> src, int target_pos, int target_pos2, int dest_pos, size_t value, size_t value2, size_t* dest, int* nextIndex) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
			return;
		}	

		int newIndex = (src.begin + index) % src.size;

		if ((src.pointer[newIndex * 3 + target_pos] == value) && (src.pointer[newIndex * 3 + target_pos2] == value2)) {
			int add = atomicAdd(nextIndex, 1);
			size_t* dest_p = (size_t*) (dest + add * 1) ;
			*dest_p = src.pointer[newIndex * 3 + dest_pos];	
		}
}


class SelectOperation : virtual public Operation
{
	private:
		std::vector<size_t> constants_;
		int arr_;
		CircularBuffer<size_t>* storePointer_;
		
	public:		
		SelectOperation(std::vector<size_t> constants, std::vector<std::string> variables, int arr) : Operation(variables) {
			this->constants_ = constants;	
			this->arr_ = arr;
		}
		
		void setStorePointer(CircularBuffer<size_t>* storePointer) {
			this->storePointer_ = storePointer;
		};

		void execute() {	
			CircularBuffer<size_t> d_storePointer = *storePointer_;
			int storeSize = d_storePointer.getLength();

			//Initialize elements	
			int* atomicIndex;
			cudaMalloc(&atomicIndex, sizeof(int));
			cudaMemset(atomicIndex, 0,  sizeof(int));
			
			int blockSize = 256;		
			int gridSize = (storeSize / blockSize) + 1;
	 		
	 		//Allocate result
			result_.allocateOnDevice(storeSize);
			
			//Select the correct kernel with specific parameters based on the position of the constants
			switch(arr_) {

				case(1): {
					unarySelect<<<gridSize,blockSize>>>(d_storePointer, 0, 1, 2, constants_[0], result_.pointer,  atomicIndex);
					break;
				}

				case(2): {
					unarySelect<<<gridSize,blockSize>>>(d_storePointer,  1, 0, 2, constants_[0], result_.pointer,  atomicIndex);
					break;
				}
					
				case(4): {
			        	unarySelect<<<gridSize,blockSize>>>(d_storePointer,  2, 0, 1, constants_[0], result_.pointer, atomicIndex);
			        	break;
				}
		
				case(3): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 0, 1, 2, constants_[0], constants_[1], result_.pointer, atomicIndex);
					break;
				}
				case(5): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 0, 2, 1, constants_[0], constants_[1], result_.pointer, atomicIndex);
					break;
				}
				case(6): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 1, 2, 0, constants_[0], constants_[1], result_.pointer, atomicIndex);
					break;
				}
					
				case(0): {
					if (d_storePointer.end < d_storePointer.begin) {
						int finalHalf = (d_storePointer.size) - d_storePointer.begin;
						cudaMemcpy(result_.pointer, d_storePointer.pointer + d_storePointer.begin * 3, finalHalf * 3 * sizeof(size_t), cudaMemcpyDeviceToDevice);			
	
						int firstHalf = storeSize - finalHalf;
						cudaMemcpy(result_.pointer + finalHalf * result_.width, d_storePointer.pointer, firstHalf * 3 * sizeof(size_t), cudaMemcpyDeviceToDevice);			
					} else {
						cudaMemcpy(result_.pointer, d_storePointer.pointer + d_storePointer.begin * 3, storeSize * 3 * sizeof(size_t), cudaMemcpyDeviceToDevice);	
					}
					cudaMemcpy(atomicIndex, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
					break;
				}
			
			}
			
			//Get actual output size	
			cudaMemcpy(&result_.height, atomicIndex, sizeof(int), cudaMemcpyDeviceToHost);
			
			//Free unused memory
			cudaFree(atomicIndex);
		}
		
		
		
	
	
};


