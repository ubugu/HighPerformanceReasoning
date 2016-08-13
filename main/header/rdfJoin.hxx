#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include "types.hxx"


//Sorter for sorting the triple due to theorder defined by the sortMask
 class TripleSorter {
	private:
		int sortMask[3];
	public:
		TripleSorter(JoinMask sortMask[3]) {
			this->sortMask[0] = static_cast<int> (sortMask[0]);
			this->sortMask[1] = static_cast<int> (sortMask[1]);
			this->sortMask[2] = static_cast<int> (sortMask[2]);
				
		}
		
		MGPU_DEVICE bool operator() (tripleContainer a, tripleContainer b) {
			int tripleA[3] = {a.subject, a.predicate, a.object};
			int tripleB[3] = {b.subject, b.predicate, b.object};
			
			if ((sortMask[0] != -1) && (tripleA[sortMask[0]] < tripleB[sortMask[0]])) {
				return true;
			}
			
			if ((sortMask[1] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] < tripleB[sortMask[1]])) {
				return true;
			}
			
			if ((sortMask[2] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] == tripleB[sortMask[1]]) && (tripleA[sortMask[2]] < tripleB[sortMask[2]])) {
				return true;
			}
			
			return false;
		}
};


//Computes inner < outer when doing join to find the elements that needs to be joined
class TripleComparator
{
	private:
		int maskA[3];
		int maskB[3];
	public:
		TripleComparator(JoinMask innerMask[3], JoinMask outerMask[3]) {
			maskA[0] = static_cast<int> (innerMask[0]);
			maskA[1] = static_cast<int> (innerMask[1]);
			maskA[2] = static_cast<int> (innerMask[2]);
			
			maskB[0] = static_cast<int> (outerMask[0]);
			maskB[1] = static_cast<int> (outerMask[1]);
			maskB[2] = static_cast<int> (outerMask[2]);			
		}
		
		MGPU_DEVICE bool operator() (tripleContainer a, tripleContainer b) {			
			int tripleA[3] = {a.subject, a.predicate, a.object};
			int tripleB[3] = {b.subject, b.predicate, b.object};

			if ((maskA[0] != -1) && (tripleA[maskA[0]] < tripleB[maskB[0]])) {
				return true;
			}
		
			if ((maskA[1] != -1) && (tripleA[maskA[0]] == tripleB[maskB[0]]) && (tripleA[maskA[1]] < tripleB[maskA[1]])) {
				return true;
			}
			
			if ((maskA[2] != -1) && (tripleA[maskA[0]] == tripleB[maskB[0]]) && (tripleA[maskA[1]] == tripleB[maskA[1]]) && (tripleA[maskA[2]] < tripleB[maskA[2]])) {
				return true;
			}
		
			return false;
		}
};



struct mask_s {
	int subject;
	int predicate;
	int object;
};
 
__global__ void reorderTriple(tripleContainer* src, tripleContainer* dest, int maxSize, mask_s mask) {
		
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}

	int triple[3] = {src[destIndex].subject, src[destIndex].predicate, src[destIndex].object};
 	tripleContainer destTriple = {triple[mask.subject], -1, -1};
 	
 	if (mask.predicate != -1) {
 		destTriple.predicate = triple[mask.predicate];
 	}
 	
 	
 	if (mask.object != -1) {
	 	destTriple.object = triple[mask.object];
 	}
 	
	dest[destIndex] = destTriple;
}

__global__ void indexCopy(tripleContainer* innerSrc, tripleContainer* innerDest, tripleContainer* outerSrc, tripleContainer* outerDest, int2* srcIndex, int maxSize) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}
	
	int innerIndex = srcIndex[destIndex].x;
	int outerIndex = srcIndex[destIndex].y;
	
	innerDest[destIndex] = innerSrc[innerIndex];	
	outerDest[destIndex] = outerSrc[outerIndex];
}


std::vector<mem_t<tripleContainer>*> rdfJoin(tripleContainer* innerTable, int innerSize, tripleContainer* outerTable, int outerSize, JoinMask innerMask[3], JoinMask outerMask[3])
{
	standard_context_t context;
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	TripleSorter* innerSorter = new TripleSorter(innerMask);
	TripleSorter* outerSorter = new TripleSorter(outerMask);
	TripleComparator* comparator = new TripleComparator(innerMask, outerMask);

	struct timeval beginCu, end;

	mask_s mask;
	mask.subject = static_cast<int> (outerMask[0]);
	mask.predicate = static_cast<int> (outerMask[1]);
	mask.object = static_cast<int> (outerMask[2]);	
	int gridSize = 64;
	int blockSize = (outerSize/ gridSize) + 1;
	mem_t<tripleContainer>* tempOuter = new mem_t<tripleContainer>(outerSize, context);
	reorderTriple<<<gridSize, blockSize>>>(outerTable, tempOuter->data(), outerSize, mask);
	cudaDeviceSynchronize();

	
	//Sort the two input array
	mergesort(innerTable, innerSize , *innerSorter, context);
	mergesort(tempOuter->data(), outerSize , *innerSorter, context);
	
	
	
	//BUG che mi costringe ad invertire inner con outer?
	mem_t<int2> joinResult = inner_join( innerTable, innerSize, tempOuter->data(), outerSize,  *innerSorter, context);
		
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
        mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	gridSize = 64;
	blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), outerTable, outerResults->data(), joinResult.data(), joinResult.size());

	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);

	cudaFree(tempOuter->data());
	return finalResults;
}


