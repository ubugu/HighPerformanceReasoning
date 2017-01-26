#pragma once

#include <cstdlib>

#include "types.hxx"
#include "operations.hxx"



//Copy elements from the inner and outer table source to the output destination, taking one element at a time
__global__ void copy(size_t* dest, size_t* innersrc,  size_t* outersrc, int innerwidth, int outerwidth, int outerSize, int outputSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= outputSize) {
		return;
	}
	
	int destindex = index * (innerwidth +  outerwidth);
	int i = 0;
	
	for (; i < innerwidth; i++) {
		dest[i + destindex] = innersrc[ (index / outerSize) * innerwidth + i];
	}

	for (int k = 0; k < outerwidth; k++) {
		dest[destindex + i + k] = outersrc[(index % outerSize) * outerwidth + k];
	}
}



/*
** Class for join operation
*/
class ProductOperation : virtual public Operation
{	
	private:
		RelationTable* innerTable_;
		RelationTable* outerTable_;
		bool isOptional = false;
		

	public:
		ProductOperation(RelationTable* innerTable, RelationTable* outerTable,  std::vector<std::string> variables) : Operation(variables) {
			this->innerTable_ = innerTable;
			this->outerTable_ = outerTable;
		}
		
		ProductOperation(RelationTable* innerTable, RelationTable* outerTable,  bool isOptional, std::vector<std::string> variables) : Operation(variables) {
			this->innerTable_ = innerTable;
			this->outerTable_ = outerTable;
			this->isOptional = isOptional;
		}
	
		//Function to execute the join function due to the size of the join.
		void execute() {
			if (isOptional && (outerTable_->height == 0 )) {
				result_.allocateOnDevice(innerTable_->height);
				cudaMemcpy(result_.pointer, innerTable_->pointer, sizeof(size_t) * innerTable_->height * innerTable_->width, cudaMemcpyDeviceToDevice);
				
			} else {
				int outputSize = innerTable_->height * outerTable_->height;
			
				//Allocate result table
				result_.allocateOnDevice(outputSize);
		

				//Copy table 
				int blockSize = 512;
				int gridSize = outputSize / blockSize + 1;
				copy<<<gridSize, blockSize>>>(result_.pointer, innerTable_->pointer, outerTable_->pointer, innerTable_->width, outerTable_->width, outerTable_->height, outputSize);

				//Free all unused memory
				cudaFree(innerTable_->pointer);
				cudaFree(outerTable_->pointer);		
			}


		}

};


