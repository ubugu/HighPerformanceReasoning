#pragma once

#include <cstdlib>

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>

#include "types.hxx"
#include "operations.hxx"


template <int N>
struct Row 
{
    unsigned int index;
    size_t element[N];
};

//Computes inner < outer when doing join to find the elements that needs to be joine
template<int joinsize>
class RowComparator
{
		
	public:	
		MGPU_DEVICE bool operator() (Row<joinsize> a, Row<joinsize> b) {
			        	
			if ((a.element[0] < b.element[0])) {
				return true;
			}
			
			if ((joinsize == 2 ) && (a.element[0] == b.element[0]) && (a.element[1] < b.element[1])) {
				return true;
			}
			
			if ((joinsize == 3) && (a.element[0] == b.element[0]) && (a.element[1] == b.element[1]) && (a.element[2] < b.element[2])) {
				return true;
			}
			
			return false;
		}
};


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




//Section for defining operation classes
class JoinOperation : public Operation
{	
	private:
		Binding** innerTable;
		Binding** outerTable;
		
		//Position in inner and outer table of the varaibles to join
		std::vector<int> innervar;
		std::vector<int> outervar;
		
		//Position in the outer table of the variables not contained in the current innertable, added after the join.
		std::vector<int> copyindex;
		

	public:
		JoinOperation(Binding** innerTable, Binding** outerTable, std::vector<int> innervar, std::vector<int> outervar, std::vector<int> copyindex, std::vector<std::string> variables) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			this->variables_ = variables;
			this->copyindex = copyindex;
			this->innervar = innervar;
			this->outervar = outervar;
		}
		
	void launcher() {
			assert(innervar.size() == outervar.size());
			
			switch(innervar.size()) {
				case(1): 
					rdfJoin<1>();
					break;

				case(2): 
					rdfJoin<2>();
					break;

					
				case(3): 
					rdfJoin<3>();
					break;
			}
		}
		
		template<int joinsize>
		void rdfJoin() {		
			using namespace mgpu;
			standard_context_t context;
			
			Row<joinsize>* innertemp;
			Row<joinsize>* outertemp;
			cudaMalloc(&innertemp, sizeof(Row<joinsize>) * (*innerTable)->height);
			cudaMalloc(&outertemp, sizeof(Row<joinsize>) * (*outerTable)->height);

 			int* innerindex;
			int* outerindex;
			cudaMalloc(&innerindex, sizeof(int) * joinsize);
			cudaMalloc(&outerindex, sizeof(int) * joinsize);
			cudaMemcpy(innerindex, &innervar[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);
			cudaMemcpy(outerindex, &outervar[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);
			reduceCopy<<<100,  ((*innerTable)->height/100) + 1>>>((*innerTable)->pointer, innertemp, (*innerTable)->width, innerindex, joinsize, (*innerTable)->height);
			reduceCopy<<<100,  ((*outerTable)->height/100) + 1>>>((*outerTable)->pointer, outertemp, (*outerTable)->width, outerindex, joinsize, (*outerTable)->height);

			RowComparator<joinsize>* sorter = new RowComparator<joinsize>();
	
			//TODO togliere params per gpu su sola1
			mergesort<launch_params_t<128, 2>>(innertemp, (*innerTable)->height, *sorter, context);
			mergesort<launch_params_t<128, 2>>(outertemp, (*outerTable)->height , *sorter, context);
					
//			mergesort(innertemp, (*innerTable)->height, *innersorter, context);
//			mergesort(outertemp, (*outerTable)->height , *outersorter, context);
	
			mem_t<int2> joinResult = inner_join<launch_params_t<128, 2>>( innertemp, (*innerTable)->height, outertemp, (*outerTable)->height,  *sorter, context);
			result_ = new Binding(variables_.size(), joinResult.size());
			
			
			int* d_copyindex;
			cudaMalloc(&d_copyindex, sizeof(int) * copyindex.size());
			cudaMemcpy(d_copyindex, &copyindex[0], sizeof(int) * copyindex.size(), cudaMemcpyHostToDevice);

			int gridsize = 50;
			int blocksize = (joinResult.size() / gridsize) + 1;			
			indexCopy<<<gridsize, blocksize>>>(result_->pointer, (*innerTable)->pointer, innertemp, (*outerTable)->pointer, outertemp, d_copyindex, (*innerTable)->width, (*outerTable)->width, copyindex.size(), joinResult.data(), joinResult.size());


			std::cout << "SIZE IS " << joinResult.size() << std::endl;		




			delete(sorter);
			cudaFree(innertemp);
			cudaFree(outertemp);
			cudaFree(d_copyindex);
			//TODO VEDERE SE CANCELLARE ANCHE INNER ED OUTER TABLE ( SE NON POSSONO SERVIRE PIU TARDI).
		}
};


