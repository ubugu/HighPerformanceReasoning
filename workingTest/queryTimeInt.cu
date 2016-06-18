#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>

using namespace mgpu;

//struct to contains a single triple with a element_t type.

template<typename element_t>
struct tripleContainer {
        element_t subject;
        element_t predicate;
        element_t object;
};

 
/**
* Enum for condition that are applied 
* to the triple, and function associated
* to them.
**/
enum compareType {LT, LEQ, EQ, GT, GEQ, NC};
MGPU_DEVICE bool compare(int a, int b,  compareType type) {
	switch(type)
	{
	
		//Less than
		case(compareType::LT):
			return a < b;
		
		//Less or equal
		case(compareType::LEQ):
			return a <= b;

		//Equal
		case(compareType::EQ):	
			return a == b;

		//Greater or equal
		case(compareType::GEQ):
			return a >= b;

		//Greater
		case(compareType::GT):
			return a > b;

		//not compare, always return true.
		case(compareType::NC):
			return true;
		
		default:
			return false;
	}
}


int separateWords(std::string inputString, std::vector<std::string> &wordVector,const char separator ) {	
	const size_t zeroIndex = 0;
	size_t splitIndex = inputString.find(separator);
	
	while (splitIndex != -1)
		{
			wordVector.push_back(inputString.substr(zeroIndex, splitIndex));	
			inputString = inputString.substr(splitIndex + 1 , inputString.length() - 1);
			splitIndex = inputString.find(separator);
		}
	
	wordVector.push_back(inputString);
	return 0;
}


/*
* Make multiple select query, with specified comparison condition,
* on a triple store. Both queries and the store are supposed to 
* be already on the device. 
* 
* @param d_selectQueries : the array in which are saved the select values
* @param d_storePointer : pointer on the device to the triple store
* @param storeSize : size of the triple store
* @param comparatorMask : array of triple of comparator that are applied to the queries
*			must be of the size of the d_selectQueries
* @return a vector of type mem_t in which are saved the query results.
*/
template<typename element_t>
std::vector<mem_t<tripleContainer<element_t>>*> rdfSelect(const std::vector<tripleContainer<element_t>*> d_selectQueries, 
		const tripleContainer<element_t>* d_storePointer,
		const int storeSize, 
		std::vector<compareType*> comparatorMask) 
{

	//Initialize elements
	int querySize =  d_selectQueries.size();
	standard_context_t context; 
	auto compact = transform_compact(storeSize, context);
	std::vector<mem_t<tripleContainer<element_t>>*> finalResults;

	//Cycling on all the queries
	for (int i = 0; i < querySize; i++) {
		//Save variable to pass to the lambda operator
		tripleContainer<element_t>* currentPointer = d_selectQueries[i];
		compareType subjectComparator = comparatorMask[i][0];
		compareType predicateComparator = comparatorMask[i][1];
		compareType objectComparator = comparatorMask[i][2];

		//Execute the select query
		int query_count = compact.upsweep([=] MGPU_DEVICE(int index) {
			bool subjectEqual = false;
			bool predicateEqual = false;
			bool objectEqual = false;

			subjectEqual = compare(d_storePointer[index].subject, currentPointer->subject, subjectComparator);
			predicateEqual = compare(d_storePointer[index].predicate, currentPointer->predicate, predicateComparator);
			objectEqual = compare(d_storePointer[index].object, currentPointer->object, objectComparator);

			return (subjectEqual && predicateEqual && objectEqual);
		});

		//Create and store queries results on device
		mem_t<tripleContainer<element_t>>* currentResult = new mem_t<tripleContainer<element_t>>(query_count, context);
		tripleContainer<element_t>* d_currentResult =  currentResult->data();

		compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
			d_currentResult[dest_index] = d_storePointer[source_index];
		});
		
		finalResults.push_back(currentResult);
	}

	return finalResults;
}


enum class JoinMask {NJ = -1, SBJ = 0, PRE = 1, OBJ = 2};

class TripleSorter
{
	private:
		int sortMask[3];
	public:
		TripleSorter(JoinMask sortMask[3]) {
			this->sortMask[0] = static_cast<int> (sortMask[0]);
			this->sortMask[1] = static_cast<int> (sortMask[1]);
			this->sortMask[2] = static_cast<int> (sortMask[2]);
				
		}
		
		MGPU_DEVICE bool operator() (tripleContainer<int> a, tripleContainer<int> b) {
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



class TripleSorter2
{
	private:
		int maskA[3];
		int maskB[3];
	public:
		TripleSorter2(JoinMask innerMask[3], JoinMask outerMask[3]) {
			this->maskA[0] = static_cast<int> (innerMask[0]);
			this->maskA[1] = static_cast<int> (innerMask[1]);
			this->maskA[2] = static_cast<int> (innerMask[2]);
			
			maskB[0] = static_cast<int> (outerMask[0]);
			maskB[1] = static_cast<int> (outerMask[1]);
			maskB[2] = static_cast<int> (outerMask[2]);			
		}
		
		MGPU_DEVICE bool operator() (tripleContainer<int> a, tripleContainer<int> b) {			
			int tripleA[3] = {a.subject, a.predicate, a.object};
			int tripleB[3] = {b.subject, b.predicate, b.object};
			
			if ((maskA[1] != -1) && (tripleA[maskA[0]] < tripleB[maskB[0]])) {
				return true;
			}
		
			if ((maskA[1] != -1) && (tripleA[maskA[0]] == tripleB[maskB[0]]) && (tripleA[1] < tripleB[1])) {
				return true;
			}
			
			if ((maskA[2] != -1) && (tripleA[0] == tripleB[0]) && (tripleA[1] == tripleB[1]) && (tripleA[2] < tripleB[2])) {
				return true;
			}

			
					
			return false;
		}
};



template<typename element_t>
__global__ void indexCopy(tripleContainer<element_t>* src, tripleContainer<element_t>* dest, int2* srcIndex, const bool x, int maxSize) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int currentIndex = 0;
	
	if (destIndex >= maxSize) {
		return;
	}
	
	if (x) {
		currentIndex = srcIndex[destIndex].x;
	} else {
		currentIndex = srcIndex[destIndex].y;
	}

	dest[currentIndex] = src[currentIndex];
}

__global__ void shrink(int2* srcIndex, int* vectorSize, int maxSize) {
	
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (destIndex >= maxSize) {
		return;
	}
	

	printf("index %i ", destIndex);
	printf("first %i ",srcIndex[destIndex].x);
	printf("second %i \n",srcIndex[destIndex + 1].x);
	
	if (srcIndex[destIndex].x != srcIndex[destIndex + 1].x ) {
		printf("aumento \n");
		*vectorSize += 1;
	} 

}

__global__ void dummy(int2* srcIndex, int* vectorSize, int maxSize) {
	printf("dummy \n");
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	printf("mi sa che non vado \n");
}

template<typename element_t>
std::vector<mem_t<tripleContainer<element_t>>*> rdfJoin(tripleContainer<element_t>* innerTable, int innerSize, tripleContainer<element_t>* outerTable, int outerSize, JoinMask innerMask[3], JoinMask outerMask[3])
{
	standard_context_t context;
	std::vector<mem_t<tripleContainer<element_t>>*> finalResults;
	
	TripleSorter* innerSorter = new TripleSorter(innerMask);
	TripleSorter* outerSorter = new TripleSorter(outerMask);
	
	//Sort the two input array
	mergesort(innerTable, innerSize , *innerSorter, context);
	mergesort(outerTable, outerSize , *outerSorter, context);
	
	TripleSorter2* comparator = new TripleSorter2(innerMask, outerMask);
	mem_t<int2> joinResult = inner_join(innerTable, innerSize, outerTable, outerSize, *comparator, context);
	
	
	std::vector<int2> prova = from_mem(joinResult);
	
	for (int i = 0; i << prova.size(); i++) {
	//	std::cout << prova[i].subject << " " << prova[i].predicate << " " << prova[i].object << std::endl;
		std::cout << prova[i].x << std::endl;

	}
	
	std::cout << "size is " << prova.size() << std::endl;
		
/*
	std::vector<int2> final = from_mem(joinResult);
	
	for (int i = 0; i < 100; i ++) {
		std::cout << "index is x:"<< final[i].x << std::endl; 
		std::cout << "index is y:"<< final[i].y << std::endl; 
	}

	
	mem_t<int> vectorSize(1, context);
	int* zeroValue = (int*) malloc(sizeof(int));
	*zeroValue = 0;
	cudaMemcpy(vectorSize.data(), zeroValue, sizeof(int), cudaMemcpyHostToDevice);
	shrink<<<1, 10 >>>(joinResult.data(), vectorSize.data(), joinResult.size());
	std::vector<int> size = from_mem(vectorSize);
	cudaMemcpy(zeroValue, vectorSize.data(), sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << *zeroValue << std::endl;
	
//	mem_t<tripleContainer<element_t>>* innerResults = new mem_t<tripleContainer<element_t>>(innerSize, context);
        mem_t<tripleContainer<element_t>>* outerResults = 0;
	mem_t<tripleContainer<element_t>>* innerResults = 0;
	/*
	
	//TODO divedere in numero diblocchi/thread corretto  
	indexCopy<<<1, blockSize>>>(innerTable, innerResults->data(), joinResult.data(), true, innerSize);


	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);

*/
	return finalResults;
}


//Section for defining operation classes

template <typename element_t>
class RelationalOperation
{
	private:
		mem_t<tripleContainer <element_t>>* result = 0;	
	public:
		mem_t<tripleContainer <element_t>>* getResult() {
			return this->result;
		};
		
		void setResult(mem_t<tripleContainer <element_t>>* result) {
			this->result = result;
		};
		
		mem_t<tripleContainer <element_t>>** getResultAddress() {
			return &result;
		}
		
};

template <typename element_t>
class JoinOperation : public RelationalOperation<element_t>
{
	
	private:
		mem_t<tripleContainer <element_t>>** innerTable;
		mem_t<tripleContainer <element_t>>** outerTable;
		JoinMask innerMask[3];
		JoinMask outerMask[3];

	public:
		JoinOperation(mem_t<tripleContainer <element_t>>** innerTable, mem_t<tripleContainer <element_t>>** outerTable, JoinMask innerMask[3], JoinMask outerMask[3]) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			std::copy(innerMask, innerMask + 3, this->innerMask);
			std::copy(outerMask, outerMask + 3, this->outerMask);
		};
			
		mem_t<tripleContainer <element_t>>** getInnerTable() {
			return this->innerTable;
		};
		
		mem_t<tripleContainer <element_t>>** getOuterTable() {
			return this->outerTable;
		};
		
		JoinMask* getInnerMask() {
			return this->innerMask;
		};
		
		JoinMask* getOuterMask() {
			return this->outerMask;
		};		
};

template<typename element_t>
class SelectOperation : public RelationalOperation<element_t>
{

	private:
		mem_t<tripleContainer <element_t>>* query;
		compareType operationMask[3];

	public:
		SelectOperation(mem_t<tripleContainer <element_t>>* query, compareType operationMask[3]) {
			this->query = query;
			std::copy(operationMask, operationMask + 3, this->operationMask);
		};
			
		mem_t<tripleContainer <element_t>>* getQuery() {
			return this->query;
		};
		
		compareType* getOperationMask() {
			return this->operationMask;
		};
};


template <typename element_t>
void queryManager(std::vector<SelectOperation<element_t>*> selectOp, std::vector<JoinOperation<element_t>*> joinOp, const tripleContainer<element_t>* d_storePointer, const int storeSize) {

	std::vector<tripleContainer<element_t>*> d_selectQueries;
	std::vector<compareType*> comparatorMask;
	
	for (int i = 0; i < selectOp.size(); i++) {
		d_selectQueries.push_back(selectOp[i]->getQuery()->data());
		comparatorMask.push_back(selectOp[i]->getOperationMask());
	}

	std::vector<mem_t<tripleContainer<element_t>>*> selectResults = rdfSelect(d_selectQueries, d_storePointer, storeSize, comparatorMask);
	
	for (int i = 0; i < selectResults.size(); i++) {
		selectOp[i]->setResult(selectResults[i]);
	}
	
	
	for (int i = 0; i < joinOp.size(); i++) {
		mem_t<tripleContainer<element_t>>* innerTable = *joinOp[i]->getInnerTable();
		mem_t<tripleContainer<element_t>>* outerTable = *joinOp[i]->getOuterTable();
		std::vector<mem_t<tripleContainer<element_t>>*>  joinResult = rdfJoin(innerTable->data(), innerTable->size(), outerTable->data(), outerTable->size(), joinOp[i]->getInnerMask(), joinOp[i]->getOuterMask());
	//	joinOp[i]->setResult(joinResult[0]);
	}
	
}

int main(int argc, char** argv) {
 
		using namespace std;
		struct timeval beginPr, beginCu, beginEx, end;
		gettimeofday(&beginPr, NULL);	
		
		using tripleElement = int;
		
		cudaDeviceReset();
		standard_context_t context;
		ifstream rdfStoreFile ("../rdfStore/rdfTimeInt.txt");
		string strInput;
		

       		
       		const int FILE_LENGHT = 302924;
              	           
                size_t rdfSize = FILE_LENGHT  * sizeof(tripleContainer<tripleElement>);
                tripleContainer<tripleElement>* h_rdfStore = (tripleContainer<tripleElement>*) malloc(rdfSize);

                //read store from rdfStore
 

                for (int i = 0; i <FILE_LENGHT ; i++) {
                	getline(rdfStoreFile,strInput);
                        std::vector<string> triple;
                        separateWords(strInput, triple, ' ');
			
           		h_rdfStore[i].subject = atoi(triple[0].c_str());
                        h_rdfStore[i].predicate = atoi(triple[1].c_str());
                        h_rdfStore[i].object = atoi(triple[2].c_str());

                }

     
         
                rdfStoreFile.close();

		std::vector<float> timeVector;                

                int N_CYCLE = 1;
		for (int i = 0; i < N_CYCLE; i++) {
			gettimeofday(&beginCu, NULL);


			tripleContainer<tripleElement>* d_storeVector;
			cudaMalloc(&d_storeVector, rdfSize);
			cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
			
	//		String queryString = "SELECT ?p ?w WHERE {  <http://example.org/int/1342> ?w  ?p. <http://example.org/int/1174> ?p ?w} ";
			
		        //set Queries (select that will be joined)
		        tripleContainer<tripleElement> h_queryVector1 = {1342, 1, 1}; 
		        tripleContainer<tripleElement> h_queryVector2 = {1174, 2 , 1};
		                
		        mem_t<tripleContainer<tripleElement>> d_queryVector1(1, context);
			cudaMemcpy(d_queryVector1.data(), &h_queryVector1, sizeof(tripleContainer<tripleElement>), cudaMemcpyHostToDevice);
		
		        mem_t<tripleContainer<tripleElement>> d_queryVector2(1, context);
			cudaMemcpy(d_queryVector2.data(), &h_queryVector2, sizeof(tripleContainer<tripleElement>), cudaMemcpyHostToDevice);
			//set select mask operation
			std::vector<tripleContainer<tripleElement>*> selectQuery;
			selectQuery.push_back(d_queryVector1.data());
			selectQuery.push_back(d_queryVector2.data());

			std::vector<compareType*> compareMask;
			compareType selectMask1[3];
		
			selectMask1[0] = compareType::EQ;
			selectMask1[1] = compareType::NC;
			selectMask1[2] = compareType::NC;

			compareMask.push_back(selectMask1);
		
			compareType selectMask2[3];		
			selectMask2[0] = compareType::EQ;
			selectMask2[1] = compareType::NC;
			selectMask2[2] = compareType::NC;
		
			compareMask.push_back(selectMask2);
		
			//set Join mask
			JoinMask innerMask[3];
			innerMask[0] = JoinMask::PRE;
			innerMask[1] = JoinMask::OBJ;
			innerMask[2] = JoinMask::NJ;
			
			JoinMask outerMask[3];
			outerMask[0] = JoinMask::OBJ;
			outerMask[1] = JoinMask::PRE;
			outerMask[2] = JoinMask::NJ;

			
			SelectOperation<tripleElement>  selectOp1(&d_queryVector1, selectMask1);
			SelectOperation<tripleElement>  selectOp2(&d_queryVector2, selectMask2);
		
			JoinOperation<tripleElement>  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), innerMask, outerMask);
		
			std::vector<SelectOperation<tripleElement>*> selectOperations;
			std::vector<JoinOperation<tripleElement>*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			selectOperations.push_back(&selectOp2);
			joinOperations.push_back(&joinOp);
		
			gettimeofday(&beginEx, NULL);	
			
		
			queryManager<tripleElement>(selectOperations, joinOperations, d_storeVector, FILE_LENGHT);
			
			
			std::vector<tripleContainer<tripleElement>> selectResults = from_mem(*selectOp1.getResult());
			std::vector<tripleContainer<tripleElement>> selectResults2 = from_mem(*selectOp2.getResult());
		/*	std::vector<tripleContainer<tripleElement>> finalResults = from_mem(*joinOp.getResult());*/
			cudaDeviceSynchronize();
			
		
			
			gettimeofday(&end, NULL);
		
			float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
			float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			
			timeVector.push_back(cuTime);
			
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;
		
			cout << "first select result" << endl;
			
			cout << selectResults.size() << endl;
			for (int i = 0; i < selectResults.size(); i++) {
				cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
			}
		
			cout << "second select result" << endl;
	
			cout << selectResults2.size() << endl;
			for (int i = 0; i < selectResults2.size(); i++) {
				cout << selectResults2[i].subject << " " << selectResults2[i].predicate << " "  << selectResults2[i].object << endl; 
			}
		
			cout << "final result" << endl;
			
		/*	cout << finalResults.size() << endl;
		/*	for (int i = 0; i < finalResults.size(); i++) {
				cout << finalResults[i].subject << " " << finalResults[i].predicate << " "  << finalResults[i].object << endl; 
			} */
			
		/*				
			cudaFree((*joinOp.getResult()).data());
			cudaFree((*selectOp1.getResult()).data());
			cudaFree((*selectOp2.getResult()).data());
			cudaFree(d_storeVector);*/
	
		}
		
		int vecSize = timeVector.size();
		float mean = 0;
		float variance = 0;

		for (int i = 0; i < vecSize; i++) {
			mean += timeVector[i];
			variance += timeVector[i] * timeVector[i];
			cout << timeVector[i] << endl;
		}
		mean = mean / ((float) vecSize);
		variance = variance / ((float) vecSize);
		variance = variance - (mean * mean);

		cout << "mean cuda time " << mean << endl;
		cout << "variance cuda time " << variance << endl;
		
		return 0;

}


