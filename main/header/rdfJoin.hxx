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
    size_t element[N];
};

//Computes inner < outer when doing join to find the elements that needs to be joined
template <int innersize, int outersize = innersize>
class RowComparator
{
	private:
		int maskA[3] = {-1, -1, -1};
		int maskB[3] = {-1, -1, -1};
	public:
		RowComparator(std::vector<int> innerMask, std::vector<int> outerMask) {
			std::copy(innerMask.begin(), innerMask.end(), maskA);
			std::copy(outerMask.begin(), outerMask.end(), maskB);
		}
		
		MGPU_DEVICE bool operator() (Row<innersize> a, Row<outersize> b) {
			        	
			if ((maskA[0] != -1) && (a.element[maskA[0]] < b.element[maskB[0]])) {
				return true;
			}
			
			if ((maskA[1] != -1) && (a.element[maskA[0]] == b.element[maskB[0]]) && (a.element[maskA[1]] < b.element[maskB[1]])) {
				return true;
			}
			
			if ((maskA[2] != -1) && (a.element[maskA[0]] == b.element[maskB[0]]) && (a.element[maskA[1]] == b.element[maskB[1]]) && (a.element[maskA[2]] < b.element[maskB[2]])) {
				return true;
			}
			
			return false;
		}
};


template<int srcsize, int destsize>
__global__ void reduceCopy(Row<srcsize>* src, Row<destsize>* dest, int* srcpos, int* destpos, int width, int maxindex) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index >= maxindex) {
		return;
	}
	for (int i = 0; i < width; i++) {
		dest[index].element[destpos[i]] = src[index].element[srcpos[i]];
	}
}


template<int innersize, int outersize>
__global__ void indexCopy(size_t* dest, Row<innersize>* innersrc, Row<outersize>* outersrc, int* indexes, int innerwidth, int outerwidth, int2* joinindex, int maxindex) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= maxindex) {
		return;
	}
	
	int destindex = index * (innerwidth + outerwidth);
	int i = 0;
	for (; i < innerwidth; i++) {
		dest[i + destindex] = innersrc[joinindex[index].x].element[i];
	}

	for (int k = 0; k < outerwidth; k++) {
		dest[destindex + i + k] = outersrc[joinindex[index].y].element[indexes[k]];
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
			std::pair<int,int> joinsize = std::make_pair((*innerTable)->width, (*outerTable)->width);
			
			switch(joinsize.first) {
				case(1): 
					{
						switch(joinsize.second) {
							case(1):
								rdfJoin<1,1>();
								break;
							
							case(2):
								rdfJoin<1,2>();
								break;							
							
							case(3):
								rdfJoin<1,3>();
								break;

						}	
					}
					break;

				case(2): 
					{
						switch(joinsize.second) {
							case(1):
								rdfJoin<2,1>();
								break;
							
							case(2):
								rdfJoin<2,2>();
								break;							
							
							case(3):
								rdfJoin<2,3>();
								break;

						}	
					}
					break;

					
				case(3): 
					{
						switch(joinsize.second) {
							case(1):
								rdfJoin<3,1>();
								break;
							
							case(2):
								rdfJoin<3,2>();
								break;							
							
							case(3):
								rdfJoin<3,3>();
								break;

						}	
					}
					break;
					
				case(4): 
					{
						switch(joinsize.second) {
							case(1):
								rdfJoin<4,1>();
								break;
							
							case(2):
								rdfJoin<4,2>();
								break;							
							
							case(3):
								rdfJoin<4,3>();
								break;

						}	
					}
					break;

					
				case(5): 
					{
						switch(joinsize.second) {
							case(1):
								rdfJoin<5,1>();
								break;
							
							case(2):
								rdfJoin<5,2>();
								break;							
							
							case(3):
								rdfJoin<5,3>();
								break;

						}	
					}
					break;					
			}
		}
		
		template<int innersize, int outersize>
		void rdfJoin() {	
			Row<innersize>* innertemp;
			Row<outersize>* outertemp;
				
			if ((sizeof(*innertemp)  == (*innerTable)->width * sizeof(size_t)) && (sizeof(*outertemp) == (*outerTable)->width * sizeof(size_t))) {
				innertemp = reinterpret_cast<Row<innersize>*>((*innerTable)->pointer);
				outertemp = reinterpret_cast<Row<outersize>*>((*outerTable)->pointer);
			} else  {
				//TODO IMPLEMENTARE
				std::cout << "ALIGNMENT IS DIFFERENT; NOT IMPLEMENTED YET " << std::endl;
				exit(-1);
			}
			
			
			using namespace mgpu;
			standard_context_t context;
			
			RowComparator<innersize>* innersorter = new RowComparator<innersize>(innervar, innervar);
			RowComparator<outersize>* outersorter = new RowComparator<outersize>(outervar, outervar);
	
			//TODO togliere params per gpu su sola1
			mergesort<launch_params_t<128, 2>>(innertemp, (*innerTable)->height, *innersorter, context);
			mergesort<launch_params_t<128, 2>>(outertemp, (*outerTable)->height , *outersorter, context);
			
//			mergesort(innertemp, (*innerTable)->height, *innersorter, context);
//			mergesort(outertemp, (*outerTable)->height , *outersorter, context);
	
			//TODO  vedere se rimuovere costraint anche di align
			//Need to made row width equal and align in the same position the join varaibles
			Row<outersize>* reducedinner;
			bool isreduced = !(innersize == outersize && innervar == outervar);
			if (!isreduced) {
				reducedinner = reinterpret_cast<Row<outersize>*>( innertemp);
			} else  {
				cudaMalloc(&reducedinner, sizeof(Row<outersize>) *  (*innerTable)->height);
			
				int* reduceindex;
				int* destpos;
				int joinsize = innervar.size();
				cudaMalloc(&reduceindex, sizeof(int) * joinsize);
				cudaMalloc(&destpos, sizeof(int) * joinsize);
				cudaMemcpy(reduceindex, &innervar[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);
				cudaMemcpy(destpos, &outervar[0], sizeof(int) * joinsize, cudaMemcpyHostToDevice);

				reduceCopy<<<20,  ((*innerTable)->height/20) + 1>>>(innertemp, reducedinner, reduceindex, destpos, joinsize, (*innerTable)->height);
	
				cudaFree(reduceindex);
				cudaFree(destpos);
			}
	
			
			mem_t<int2> joinResult = inner_join<launch_params_t<128, 2>>( reducedinner, (*innerTable)->height, outertemp, (*outerTable)->height,  *outersorter, context);
			result_ = new Binding(variables_.size(), joinResult.size());
			
			int* d_copyindex;
			cudaMalloc(&d_copyindex, sizeof(int) * copyindex.size());
			cudaMemcpy(d_copyindex, &copyindex[0], sizeof(int) * copyindex.size(), cudaMemcpyHostToDevice);

			int gridsize = 10;
			int blocksize = (joinResult.size() / gridsize) + 1;			
			indexCopy<<<gridsize, blocksize>>>(result_->pointer, innertemp, outertemp, d_copyindex, (*innerTable)->width, copyindex.size(), joinResult.data(), joinResult.size());


			delete(innersorter);
			delete(outersorter);
			cudaFree(innertemp);
			cudaFree(outertemp);
			cudaFree(d_copyindex);
			if (isreduced) {
				cudaFree(reducedinner);			
			}
			
			//TODO VEDERE SE CANCELLARE ANCHE INNER ED OUTER TABLE ( SE NON POSSONO SERVIRE PIU TARDI).
		}
};


