#pragma once

#include <cstdlib>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>

#include "types.hxx"
#include "operations.hxx"
#include "rdfJoin.hxx"

//Copy elements from the inner and outer table source to the output destination, using an additional boolean vector indicating that the element has been copied
template<int joinsize>
__global__ void indexCopy(size_t* dest, size_t* innersrc, Row<joinsize>* innertemp, size_t* outersrc, Row<joinsize>* outertemp, int* indexes, int innerwidth, int outerwidth, int indexsize, int2* joinindex, int maxindex, int* bit_arr) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= maxindex) {
		return;
	}
	
	int destindex = index * (innerwidth + indexsize);
	int i = 0;
	for (; i < innerwidth; i++) {
		dest[i + destindex] = innersrc[innertemp[joinindex[index].x].index * innerwidth + i];
	}

	for (int k = 0; k < indexsize; k++) {
		dest[destindex + i + k] = outersrc[outertemp[joinindex[index].y].index * outerwidth + indexes[k]];
	}
	
	bit_arr[innertemp[joinindex[index].x].index] = 0;
}

//Copy operations for the elements that dind't find a match in the join
__global__ void copysweep(size_t* dest, size_t* src, int destwidth, int srcwidth, int* bit_arr, int* prefix_arr,  int maxValue ) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= maxValue) {
		return; 
	}
	

	if (bit_arr[index] != 0) {
		for (int i = 0; i < srcwidth; i++) {
			dest[i + destwidth * (prefix_arr[index] - 1)] = src[index * srcwidth + i];		
		}
	}
}


/*
** Class for optional operation
*/
class OptionalOperation : virtual public Operation
{	
	private:
		RelationTable* innerTable_;
		RelationTable* outerTable_;
		
		//Position in inner and outer table of the varaibles to join
		std::vector<int> innerVar_;
		std::vector<int> outerVar_;
		
		//Position in the outer table of the variables not contained in the current innertable, added after the join.
		std::vector<int> copyindex_;
		

	public:
		OptionalOperation(RelationTable* innerTable, RelationTable* outerTable, std::vector<int> innervar, std::vector<int> outervar, std::vector<int> copyindex, std::vector<std::string> variables)  : Operation(variables) {
			this->innerTable_ = innerTable;
			this->outerTable_ = outerTable;
			this->copyindex_ = copyindex;
			this->innerVar_ = innervar;
			this->outerVar_ = outervar;
		}
		
		//Function to execute the optional function due to the size of the join.
		void execute() {
			assert(innerVar_.size() == outerVar_.size());
			switch(innerVar_.size()) {
				case(1): 
					rdfOptional<1>();
					break;
				case(2): 
					rdfOptional<2>();
					break;
				case(3): 
					rdfOptional<3>();
					break;
				case(4): 
					rdfOptional<4>();
					break;
				case(5): 
					rdfOptional<5>();
					break;				
				case(6): 
					rdfOptional<6>();
					break;				
				case(7): 
					rdfOptional<7>();
					break;				
				case(8): 
					rdfOptional<8>();
					break;				
				case(9): 
					rdfOptional<9>();
					break;				
				case(10): 
					rdfOptional<10>();
					break;				
			}
		}
		
		//Function for the optional operation
		template<int joinsize>
		void rdfOptional() {		
			using namespace mgpu;
			standard_context_t context;
			
			//Allocate temporary tables
			Row<joinsize>* innertemp;
			Row<joinsize>* outertemp;
			cudaMalloc(&innertemp, sizeof(Row<joinsize>) * innerTable_->height);
			cudaMalloc(&outertemp, sizeof(Row<joinsize>) * outerTable_->height);
			
			//Allocate indexes on the device
 			int* innerindex;
			int* outerindex;
			cudaMalloc(&innerindex, sizeof(int) * joinsize);
			cudaMalloc(&outerindex, sizeof(int) * joinsize);
			cudaMemcpy(innerindex, &innerVar_[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);
			cudaMemcpy(outerindex, &outerVar_[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);
			
			//Copy elements from the original tables to the reduced ones
			int blockSize = 256;
			int gridSize = innerTable_->height / blockSize + 1;	
			reduceCopy<<<gridSize,  blockSize>>>(innerTable_->pointer, innertemp, innerTable_->width, innerindex, joinsize, innerTable_->height);
			gridSize = outerTable_->height / blockSize + 1;
			reduceCopy<<<gridSize,  blockSize>>>(outerTable_->pointer, outertemp, outerTable_->width, outerindex, joinsize, outerTable_->height);

			//Sort and join elements
			RowComparator<joinsize>* sorter = new RowComparator<joinsize>();
			mergesort<launch_params_t<256, 1>>(innertemp, innerTable_->height, *sorter, context);
			mergesort<launch_params_t<256, 1>>(outertemp, outerTable_->height , *sorter, context);
			mem_t<int2> joinResult = inner_join<launch_params_t<128, 3>>( innertemp, innerTable_->height, outertemp, outerTable_->height,  *sorter, context);
			
			//Allocate result table, setting all values to 0
			result_.allocateOnDevice(joinResult.size() + innerTable_->height);
			cudaMemset(result_.pointer + joinResult.size() * result_.width, 0, innerTable_->height * result_.width * sizeof(size_t));
			
			//Allocate arrays for the prefix sum
			thrust::device_vector<int> bitArr(innerTable_->height);
			thrust::device_vector<int> prefixArr(innerTable_->height);
			thrust::fill(bitArr.begin(), bitArr.end(), 1);

			//Copies element from the outer table and set the isCopied boolean value
			int* d_copyindex;
			cudaMalloc(&d_copyindex, sizeof(int) * copyindex_.size());
			cudaMemcpy(d_copyindex, &copyindex_[0], sizeof(int) * copyindex_.size(), cudaMemcpyHostToDevice);

			//Copy elements from the input tables to the output one
			gridSize = (joinResult.size() / gridSize) + 1;			
			indexCopy<<<gridSize, blockSize>>>(result_.pointer, innerTable_->pointer, innertemp, outerTable_->pointer, outertemp, d_copyindex, innerTable_->width, outerTable_->width, copyindex_.size(), joinResult.data(), joinResult.size(), bitArr.data().get());
			
			//Prefix sum operation on the isCopied boolean array
			thrust::inclusive_scan(bitArr.begin(), bitArr.end(), prefixArr.begin());
			
			//Copy of the elements that dind't take part into the join operation
			gridSize = (innerTable_->height / gridSize) + 1;
			copysweep<<<gridSize,blockSize>>>(result_.pointer + joinResult.size() * result_.width, innerTable_->pointer, result_.width, innerTable_->width,  bitArr.data().get(), prefixArr.data().get(), innerTable_->height);
				
			//Update size of the result
			cudaMemcpy(&(result_.height), prefixArr.data().get() + innerTable_->height - 1, sizeof(int), cudaMemcpyDeviceToHost);
			result_.height += joinResult.size();
			
			//Free unused memory
			delete(sorter);
			cudaFree(innertemp);
			cudaFree(outertemp);
			cudaFree(innerindex);
			cudaFree(outerindex);
			cudaFree(d_copyindex);
			cudaFree(innerTable_->pointer);
			cudaFree(outerTable_->pointer);

		}
		
		

	
};


