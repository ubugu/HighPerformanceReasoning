#pragma once

#include <cstdlib>

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>

#include "types.hxx"
#include "operations.hxx"


//VARAIBLES FOR JOIN OPERATIONS
template <int size>
struct Row 
{
    unsigned int index;
    size_t element[size];
};

//Comparator function for compargine elements of types ROW with different joinsize
template<int joinsize>
class RowComparator
{
	public:	
		MGPU_DEVICE bool operator() (Row<joinsize> a, Row<joinsize> b) {
			
			int k = 0;
			bool result = false;
			for (int i = 1; i <= joinsize; i++) {
				for (k = 0; k < i; k++) {
					if (k == (i - 1)) {
						result 	= (a.element[k] < b.element[k]);
						break;
					} else {
						result = (a.element[k] == b.element[k]);
					}
					
					if (!result) 
						break;
				}
				
				if (result) {
					return true;
				}
			}
			
			return result;
		}
};

//Copy elements from the source relational table to the destination Row structure
template<int destsize>
__global__ void reduceCopy(size_t* src, Row<destsize>* dest, int srcwidth, int* srcpos, int joinsize, int maxindex) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= maxindex) {
		return;
	}

	for (int i = 0; i < joinsize; i++) {
		dest[index].element[i] = src[srcpos[i] + index * srcwidth];
	}
	dest[index].index = index;
}


//Copy elements from the inner and outer table source to the output destination
template<int joinsize>
__global__ void indexCopy(size_t* dest, size_t* innersrc, Row<joinsize>* innertemp, size_t* outersrc, Row<joinsize>* outertemp, int* indexes, int innerwidth, int outerwidth, int indexsize, int2* joinindex, int maxindex) {
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
}



/*
** Class for join operation
*/
class JoinOperation : virtual public Operation
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
		JoinOperation(RelationTable* innerTable, RelationTable* outerTable, std::vector<int> innervar, std::vector<int> outervar, std::vector<int> copyindex, std::vector<std::string> variables) : Operation(variables) {
			this->innerTable_ = innerTable;
			this->outerTable_ = outerTable;
			this->copyindex_ = copyindex;
			this->innerVar_ = innervar;
			this->outerVar_ = outervar;
		}
	
		//Function to execute the join function due to the size of the join.
		void execute() {
			assert(innerVar_.size() == outerVar_.size());
	
			switch(innerVar_.size()) {
				case(1): 
					rdfJoin<1>();
					break;
				case(2): 
					rdfJoin<2>();
					break;
				case(3): 
					rdfJoin<3>();
					break;
				case(4): 
					rdfJoin<4>();
					break;
				case(5): 
					rdfJoin<5>();
					break;				
				case(6): 
					rdfJoin<6>();
					break;				
				case(7): 
					rdfJoin<7>();
					break;				
				case(8): 
					rdfJoin<8>();
					break;				
				case(9): 
					rdfJoin<9>();
					break;				
				case(10): 
					rdfJoin<10>();
					break;				
			}
		}
	
		//Function for the join operation
		template<int joinsize>
		void rdfJoin() {		
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

			//Allocate result table
			result_.allocateOnDevice(joinResult.size());

			//Copy indexex for the right table
			int* d_copyindex;
			cudaMalloc(&d_copyindex, sizeof(int) * copyindex_.size());
			cudaMemcpy(d_copyindex, &copyindex_[0], sizeof(int) * copyindex_.size(), cudaMemcpyHostToDevice);

			//Copy elements from the input tables to the output one
			gridSize  = (joinResult.size() /blockSize) + 1;	
			indexCopy<<<gridSize, blockSize>>>(result_.pointer, innerTable_->pointer, innertemp, outerTable_->pointer, outertemp, d_copyindex, innerTable_->width, outerTable_->width, copyindex_.size(), joinResult.data(), joinResult.size());

			//Free all unused memory
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


