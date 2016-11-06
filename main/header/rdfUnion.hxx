#pragma once

#include <cstdlib>

#include "types.hxx"
#include "operations.hxx"

//Copy the element of the left table in order
__global__ void firstCopy(size_t* dest, size_t* src, int srcwidth, int destwidth, int heigth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= heigth) {
		return;
	}	

	for (int i =0; i < srcwidth; i++) {
		dest[index * destwidth + i] = src[index * srcwidth + i];
	}
}

//Copy the element of the right table due to the index provided
__global__ void secondCopy(size_t* dest, size_t* src, int* unionIndex, int srcwidth, int destwidth, int heigth) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= heigth) {
		return;
	}	
	
	for (int i =0; i < srcwidth; i++) {
		dest[index * destwidth + unionIndex[i]] = src[index * srcwidth + i];
	}
}
	 
/**
** Class for Union operation
**/
class UnionOperation : public Operation
{
	private:
		std::vector<int> unionIndex_;  	//Index indicating the position in which copy the element of the right table
		RelationTable* first_;		//Left  table
		RelationTable* second_;		//Right table
		bool sameHeader_ = true;		//Bool value to indicate which union make
		
	public:
		
		UnionOperation(RelationTable* first, RelationTable* second, std::vector<int> unionIndex, std::vector<std::string> variables) : Operation(variables) {
			this->first_ = first;
			this->second_ = second;
			this->unionIndex_ = unionIndex;
			
			//Check if the two tables have the same header
			for (int i = 0; i <unionIndex_.size(); i++) {
				sameHeader_ &= (unionIndex_[i] == i);
				if (!sameHeader_)
					break;	
			}
		}

		void execute() {
			//Allocate result
			result_.allocateOnDevice(first_->height + second_->height);
			
			/*
			**Execute the union by xopying the two input table into the output one. 
			**If the two table have the same header the cudaMemcpy function is used
			** otherwise two specialised function are used-
			*/
			if (sameHeader_) {
				cudaMemcpy(result_.pointer, first_->pointer, sizeof(size_t) * first_->height * first_->width, cudaMemcpyDeviceToDevice);				
				cudaMemcpy(result_.pointer + (first_->height * first_->width), second_->pointer, sizeof(size_t) * second_->height * second_->width, cudaMemcpyDeviceToDevice);
			} else {
				cudaMemset(result_.pointer, 0, (first_->height + second_->height) * result_.width * sizeof(size_t));
				
				//Load union index on device
				int* d_index;
				cudaMalloc(&d_index, unionIndex_.size() * sizeof(int));		
				cudaMemcpy(d_index, &unionIndex_[0], unionIndex_.size() * sizeof(int), cudaMemcpyHostToDevice);

				//Launch first copy
				int blockSize = 256;
				int gridSize = first_->height / blockSize + 1;	
				firstCopy<<<gridSize, blockSize>>>(result_.pointer, first_->pointer, first_->width, result_.width, first_->height);
			
				//Launch second copy
				gridSize = second_->height / blockSize + 1;
				secondCopy<<<gridSize, blockSize>>>((result_.pointer + first_->height * result_.width),  second_->pointer, d_index, second_->width, result_.width, second_->height);
			
				//Free memory
				cudaFree(first_->pointer);
				cudaFree(second_->pointer);
				cudaFree(d_index);
			}
			
		}
		

	
};


