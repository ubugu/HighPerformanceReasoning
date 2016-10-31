#pragma once

#include <cstdlib>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "types.hxx"
#include "operations.hxx"

enum class SelectArr { SPO = 0, S = 1, P = 2, O = 4, SP = 3, SO = 5, PO = 6, };

//Select kernel for one constant select
__global__ void unarySelect (CircularBuffer<size_t> src, int target_pos, size_t value, thrust::detail::normal_iterator<thrust::device_ptr<int>> bit_array) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;

	if (src.pointer[newIndex * 3 + target_pos] == value) {
		bit_array[index] = 1;
	}
}

//Select kernel for two constant select
__global__ void binarySelect (CircularBuffer<size_t> src, int target_pos, int target_pos2, size_t value, size_t value2, thrust::detail::normal_iterator<thrust::device_ptr<int>> bit_array) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
			return;
		}	
		
		int newIndex = (src.begin + index) % src.size;

		if ((src.pointer[newIndex * 3 + target_pos] == value) && (src.pointer[newIndex * 3 + target_pos2] == value2)) {
			bit_array[index] = 1;
		}
}

//Copy from store using the two array of prefix sum
__global__ void doubleCopysweep(size_t* dest, CircularBuffer<size_t> src, int* prefix_arr, int* bit_arr, int first, int second, int maxValue ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= maxValue) {
		return;
	}
	

	int newIndex = (src.begin + index) % src.size;
	
	if (bit_arr[index] != 0) {
		dest[2 * (prefix_arr[index] - 1)] =  src.pointer[newIndex * 3 + first];
		dest[2 * (prefix_arr[index] - 1) + 1] =  src.pointer[newIndex * 3 + second];
	}
}


//Copy from store using the two array of prefix sum
__global__ void singleCopysweep(size_t* dest, CircularBuffer<size_t> src, int* prefix_arr, int* bit_arr, int position, int maxValue ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= maxValue) {
		return;
	}
	

	int newIndex = (src.begin + index) % src.size;
	
	if (bit_arr[index] != 0) {
		dest[(prefix_arr[index] - 1)] =  src.pointer[newIndex * 3 + position];
	}
}


class SelectOperation : public Operation
{
	private:
		std::vector<size_t> constants_;
		int arr_;
		CircularBuffer<size_t>* storePointer_;

	public:
		SelectOperation(std::vector<size_t> constants, std::vector<std::string> variables, int arr) : Operation(variables){
			this->constants_ = constants;	
			this->arr_ = arr;
		}
		
		void setStorePointer(CircularBuffer<size_t>* storePointer) {
			this->storePointer_ = storePointer;
		};
		
		void execute() {
			CircularBuffer<size_t> d_storePointer = *storePointer_;
			int storeSize = d_storePointer.getLength();
			
			int blockSize = 256;		
			int gridSize = (storeSize / blockSize) + 1;
			
	 		//Allocate result
			result_.allocateOnDevice(storeSize);
			
			//Initialize elements	
			thrust::device_vector<int> bitArr(storeSize);
			thrust::device_vector<int> prefixArr(storeSize);
 			cudaMemset(bitArr.data().get(), 0, storeSize * sizeof(int));

			switch(arr_) {

				case(1): {
					unarySelect<<<gridSize,blockSize>>>(d_storePointer, 0,constants_[0], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					doubleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(), 1, 2, storeSize);					
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
					break;
				}

				case(2): {
					unarySelect<<<gridSize,blockSize>>>(d_storePointer,  1,  constants_[0], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					doubleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(), 0, 2, storeSize);
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
					break;
				}
					
				case(4): {
			        	unarySelect<<<gridSize,blockSize>>>(d_storePointer,  2,  constants_[0], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					doubleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(), 0, 1, storeSize);
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
					break;
				}
		
				case(3): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 0, 1, constants_[0],  constants_[1], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					singleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(),2, storeSize);
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
					break;
				}
				case(5): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 0, 2, constants_[0],  constants_[1], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					singleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(), 1, storeSize);
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
					break;
				}
				
				case(6): {
					binarySelect<<<gridSize,blockSize>>>(d_storePointer, 1, 2, constants_[0],  constants_[1], bitArr.begin());
					thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
					singleCopysweep<<<gridSize,blockSize>>>(result_.pointer, d_storePointer, prefixArr.data().get(), bitArr.data().get(), 0, storeSize);
					cudaMemcpy(&(result_.height), prefixArr.data().get() + storeSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
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
					result_.height = storeSize;
					break;
				}
				
				
		
			}
			
		}
	
};


