#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>

using namespace mgpu;

//struct to contains a single triple with a element_t type.

const int MAX_LENGHT = 100;

struct tripleContainer {
        char subject[MAX_LENGHT];
        char predicate[MAX_LENGHT]  ;
        char object[MAX_LENGHT];
};


__device__ int strcasecmp_d(const char *s1, const char *s2)
{
          int c1, c2;
  
          do {
                  c1 = *s1++;
                  c2 = *s2++;
          } while (c1 == c2 && c1 != 0);
          return c1 - c2;
 }
 
 /**
* Enum for condition that are applied 
* to the triple, and function associated
* to them.
**/
enum class CompareType {LT, LEQ, EQ, GT, GEQ, NC};

__device__ bool isGreater(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return strcasecmp_d(a, b) > 0;
}

__device__ bool isGreaterEq(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return strcasecmp_d(a, b) >= 0;
}

__device__ bool isEqual(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return strcasecmp_d(a, b) == 0;
}

__device__ bool isLess(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return strcasecmp_d(a, b) < 0;
}

__device__ bool isLessEq(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return strcasecmp_d(a, b) <= 0;
}

__device__ bool notCompare(const char a[MAX_LENGHT], const char b[MAX_LENGHT]) {
	return true;
}

typedef bool (*select_func) (const char[MAX_LENGHT], const char[MAX_LENGHT]);
__device__ select_func funcs[6] = {isLess, isLessEq, isEqual, isGreater, isGreaterEq, notCompare};


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
std::vector<mem_t<tripleContainer>*> rdfSelect(const std::vector<tripleContainer*> d_selectQueries, 
		const tripleContainer* d_storePointer,
		const int storeSize, 
		std::vector<int*> comparatorMask) 
{

	//Initialize elements
	int querySize =  d_selectQueries.size();
	standard_context_t context; 
	auto compact = transform_compact(storeSize, context);
	std::vector<mem_t<tripleContainer>*> finalResults;

	//Cycling on all the queries
	for (int i = 0; i < querySize; i++) {
		//Save variable to pass to the lambda operator
		tripleContainer* currentPointer = d_selectQueries[i];
		int subjectComparator = comparatorMask[i][0];
		int predicateComparator = comparatorMask[i][1];
		int objectComparator = comparatorMask[i][2];

		//Execute the select query
		int query_count = compact.upsweep([=] MGPU_DEVICE(int index) {
			bool subjectEqual = false;
			bool predicateEqual = false;
			bool objectEqual = false;
						
			subjectEqual = funcs[subjectComparator](d_storePointer[index].subject, currentPointer->subject);
			predicateEqual = funcs[predicateComparator](d_storePointer[index].predicate, currentPointer->predicate);
			objectEqual = funcs[objectComparator](d_storePointer[index].object, currentPointer->object);

			return (subjectEqual && predicateEqual && objectEqual);
		});

		//Create and store queries results on device
		mem_t<tripleContainer>* currentResult = new mem_t<tripleContainer>(query_count, context);
		tripleContainer* d_currentResult =  currentResult->data();

		compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
			d_currentResult[dest_index] = d_storePointer[source_index];
		});
		
		finalResults.push_back(currentResult);
	}

	return finalResults;
}


/*
* Join enum to define order and which element to join
* NJ indicates a non-join value, so it is ignored during join and sorting
* So that it improves performance avoiding uneecessary conditional expression
*/
enum class JoinMask {NJ = -1, SBJ = 0, PRE = 1, OBJ = 2};

//Sorter for sorting the triple due to theorder defined by the sortMask
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
		
		MGPU_DEVICE bool operator() (tripleContainer a, tripleContainer b) {
			char* tripleA[3] = {a.subject, a.predicate, a.object};
			char* tripleB[3] = {b.subject, b.predicate, b.object};
			
			int firstComp = strcasecmp_d(tripleA[sortMask[0]], tripleB[sortMask[0]]);
			if ((sortMask[0] != -1) && (firstComp < 0)) {
				return true;
			}
			
			int secondComp = 0;
			if ((sortMask[1] != -1)){
				secondComp = strcasecmp_d(tripleA[sortMask[1]], tripleB[sortMask[1]]);
				if ((firstComp == 0) && (secondComp < 0)) {
					return true;
				}
			}
			
			if ((sortMask[2] != -1)){
				int thirdComp = strcasecmp_d(tripleA[sortMask[2]], tripleB[sortMask[2]]);
				if ((sortMask[2] != -1) && (firstComp == 0) && (secondComp == 0) && (thirdComp < 0)) {
					return true;
				}
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
			char* tripleA[3] = {a.subject, a.predicate, a.object};
			char* tripleB[3] = {b.subject, b.predicate, b.object};
			
			int firstComp = strcasecmp_d(tripleA[maskA[0]], tripleB[maskB[0]]);
			if ((maskA[0] != -1) && (firstComp < 0)) {
				return true;
			}
			
			int secondComp = 0;
			if ((maskA[1] != -1)){
				secondComp = strcasecmp_d(tripleA[maskA[1]], tripleB[maskB[1]]);
				if ((firstComp == 0) && (secondComp < 0)) {
					return true;
				}
			}
			
			if ((maskA[2] != -1)){
				int thirdComp = strcasecmp_d(tripleA[maskA[2]], tripleB[maskB[2]]);
				if ((maskA[2] != -1) && (firstComp == 0) && (secondComp == 0) && (thirdComp < 0)) {
					return true;
				}
			}
			return false;
			
		}
};

__global__ void indexCopy(tripleContainer* innerSrc, tripleContainer* innerDest, tripleContainer* outerSrc, tripleContainer* outerDest, int2* srcIndex, int maxSize) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}
	
	//INVETERD INDEX DUE TO INVERTED JOIN PROBLEM (it should be inner = x, outer = y)
	int innerIndex = srcIndex[destIndex].y;
	int outerIndex = srcIndex[destIndex].x;
	
	innerDest[destIndex] = innerSrc[innerIndex];	
	outerDest[destIndex] = outerSrc[outerIndex];
}


std::vector<mem_t<tripleContainer>*> rdfJoin(tripleContainer* innerTable, int innerSize, tripleContainer* outerTable, int outerSize, JoinMask innerMask[3], JoinMask outerMask[3])
{
	standard_context_t context;
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	TripleSorter* innerSorter = new TripleSorter(innerMask);
	TripleSorter* outerSorter = new TripleSorter(outerMask);
	
	//Sort the two input array
	mergesort<launch_params_t<128, 1> >(innerTable, innerSize , *innerSorter, context);
	mergesort<launch_params_t<128, 1> >(outerTable, outerSize , *outerSorter, context);
	
	TripleComparator* comparator = new TripleComparator(innerMask, outerMask);
	
	//BUG che mi costringe ad invertire inner con outer?
	mem_t<int2> joinResult = inner_join<launch_params_t<128, 1> >(outerTable, outerSize, innerTable, innerSize,  *comparator, context);
		
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
        mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	int gridSize = 64;
	int blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), outerTable, outerResults->data(), joinResult.data(), joinResult.size());

	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);

	return finalResults;
}


//Section for defining operation classes
class JoinOperation 
{
	
	private:
		mem_t<tripleContainer>** innerTable;
		mem_t<tripleContainer>** outerTable;
		mem_t<tripleContainer>* innerResult = 0;
		mem_t<tripleContainer>* outerResult = 0;
		
		JoinMask innerMask[3];
		JoinMask outerMask[3];

	public:
		JoinOperation(mem_t<tripleContainer>** innerTable, mem_t<tripleContainer>** outerTable, JoinMask innerMask[3], JoinMask outerMask[3]) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			std::copy(innerMask, innerMask + 3, this->innerMask);
			std::copy(outerMask, outerMask + 3, this->outerMask);
		};
			
		mem_t<tripleContainer>** getInnerTable() {
			return this->innerTable;
		};
		
		mem_t<tripleContainer>** getOuterTable() {
			return this->outerTable;
		};
		
		JoinMask* getInnerMask() {
			return this->innerMask;
		};
		
		JoinMask* getOuterMask() {
			return this->outerMask;
		};
		
		mem_t<tripleContainer>* getInnerResult() {
			return this->innerResult;
		};
		
		void setInnerResult(mem_t<tripleContainer>* result) {
			this->innerResult = result;
		};
		
		mem_t<tripleContainer>** getInnerResultAddress() {
			return &innerResult;
		}
		
		mem_t<tripleContainer>* getOuterResult() {
			return this->outerResult;
		};
		
		void setOuterResult(mem_t<tripleContainer>* result) {
			this->outerResult = result;
		};
		
		mem_t<tripleContainer>** getOuterResultAddress() {
			return &outerResult;
		}			
};

class SelectOperation 
{

	private:
		mem_t<tripleContainer>* query;
		mem_t<tripleContainer>* result = 0;
		int operationMask[3];

	public:
		SelectOperation(mem_t<tripleContainer>* query, CompareType operationMask[3]) {
			this->query = query;
			this->operationMask[0] = static_cast<int> (operationMask[0]);
			this->operationMask[1] = static_cast<int> (operationMask[1]);
			this->operationMask[2] = static_cast<int> (operationMask[2]);		
		};
			
		mem_t<tripleContainer>* getQuery() {
			return this->query;
		};
		
		int* getOperationMask() {
			return this->operationMask;
		};
		
		mem_t<tripleContainer>* getResult() {
			return this->result;
		};
		
		void setResult(mem_t<tripleContainer>* result) {
			this->result = result;
		};
		
		mem_t<tripleContainer>** getResultAddress() {
			return &result;
		}
};

/**
* Function for managing query execution
**/
void queryManager(std::vector<SelectOperation*> selectOp, std::vector<JoinOperation*> joinOp, const tripleContainer* d_storePointer, const int storeSize) {

	std::vector<tripleContainer*> d_selectQueries;
	std::vector<int*> comparatorMask;
	
	for (int i = 0; i < selectOp.size(); i++) {
		d_selectQueries.push_back(selectOp[i]->getQuery()->data());
		comparatorMask.push_back(selectOp[i]->getOperationMask());
	}

	std::vector<mem_t<tripleContainer>*> selectResults = rdfSelect(d_selectQueries, d_storePointer, storeSize, comparatorMask);

		
	for (int i = 0; i < selectResults.size(); i++) {
		selectOp[i]->setResult(selectResults[i]);
	}
	
	
	for (int i = 0; i < joinOp.size(); i++) {
		mem_t<tripleContainer>* innerTable = *joinOp[i]->getInnerTable();
		mem_t<tripleContainer>* outerTable = *joinOp[i]->getOuterTable();
		std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin(innerTable->data(), innerTable->size(), outerTable->data(), outerTable->size(), joinOp[i]->getInnerMask(), joinOp[i]->getOuterMask());
		joinOp[i]->setInnerResult(joinResult[0]);
		joinOp[i]->setOuterResult(joinResult[1]);				
	}
	
}

int main(int argc, char** argv) {
               using namespace std;
                struct timeval beginPr, beginCu, beginEx, end;
                gettimeofday(&beginPr, NULL);
                cudaDeviceReset();
                standard_context_t context;

                if (argc < 2) {
                        cout << "wrong number of elements" << endl;
                }

                ifstream rdfStoreFile (argv[1]);
         //       ofstream output (argv[2]);

                string strInput;

                int fileLength = 0;
                while (std::getline(rdfStoreFile, strInput)) {
                        ++fileLength;
                }

                rdfStoreFile.clear();
                rdfStoreFile.seekg(0, ios::beg);

                size_t rdfSize = fileLength  * sizeof(tripleContainer);
                tripleContainer* h_rdfStore = (tripleContainer*) malloc(rdfSize);

		int size = 0;		
		char emptyBuff[MAX_LENGHT] = {0};
	
                for (int i = 0; i < fileLength; i++) {
                        getline(rdfStoreFile,strInput);
			std::vector<string> triple ;
                        separateWords(strInput, triple, ' ');
                        
           
                        size = triple[0].size();
                        strncpy(h_rdfStore[i].subject, emptyBuff, MAX_LENGHT);
                        strncpy(h_rdfStore[i].subject, triple[0].c_str(), size);
                     
        
                        size = triple[1].size();
                        strncpy(h_rdfStore[i].predicate, emptyBuff, MAX_LENGHT);
                        strncpy(h_rdfStore[i].predicate, triple[1].c_str(), size);
                        
                        size = triple[2].size();
                        strncpy(h_rdfStore[i].object, emptyBuff, MAX_LENGHT);
                        strncpy(h_rdfStore[i].object, triple[2].c_str(), size);
                       
                }
                
                rdfStoreFile.close();

		std::vector<float> timeCuVector;                
		std::vector<float> timeExVector;
                int N_CYCLE = 100;
		for (int i = 0; i < N_CYCLE; i++) {

                        string current = "<http://example.org/string/longlong/element/int/" + to_string(99 - i) + ">";	
                        string cicle   = "<http://example.org/string/longlong/element/int/" + to_string(i) +  ">";
			const char* str = current.c_str();
			const char* str2 = cicle.c_str();	
			
			
			gettimeofday(&beginCu, NULL);

			tripleContainer* d_storeVector;
			cudaMalloc(&d_storeVector, rdfSize);
			cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
	
			//Use query "SELECT * WHERE {  ?s ?p  <http://example.org/int/1>.  <http://example.org/int/0> ?p  ?o} ";
			
		        //set Queries (select that will be joined)
		        tripleContainer h_queryVector1;  
		        tripleContainer h_queryVector2;
		        
		        char object1[MAX_LENGHT] = {0};  
		     	std::copy(str, str + current.size(), object1);
		        
		     	strncpy(h_queryVector1.subject, emptyBuff, MAX_LENGHT);      	
		     	strncpy(h_queryVector1.predicate, emptyBuff, MAX_LENGHT);
		     	strncpy(h_queryVector1.object, object1, MAX_LENGHT);
		     	
		     	char subject2[MAX_LENGHT] = {0};  
		     	std::copy(str2, str2 + cicle.size(), subject2);
		     	
		        strncpy(h_queryVector2.subject, subject2, MAX_LENGHT);      	
		     	strncpy(h_queryVector2.predicate, emptyBuff, MAX_LENGHT);
		     	strncpy(h_queryVector2.object, emptyBuff, MAX_LENGHT);


		 	cout << "query is " << h_queryVector1.object << " " << h_queryVector2.subject << endl;
			              
		        mem_t<tripleContainer> d_queryVector1(1, context);
			cudaMemcpy(d_queryVector1.data(), &h_queryVector1, sizeof(tripleContainer), cudaMemcpyHostToDevice);
		
		        mem_t<tripleContainer> d_queryVector2(1, context);
			cudaMemcpy(d_queryVector2.data(), &h_queryVector2, sizeof(tripleContainer), cudaMemcpyHostToDevice);
			//set select mask operation
			
			std::vector<tripleContainer*> selectQuery;
			selectQuery.push_back(d_queryVector1.data());
			selectQuery.push_back(d_queryVector2.data());

			std::vector<CompareType*> compareMask;
			CompareType selectMask1[3];
		
			selectMask1[0] = CompareType::NC;
			selectMask1[1] = CompareType::NC;
			selectMask1[2] = CompareType::EQ;

			compareMask.push_back(selectMask1);
		
			CompareType selectMask2[3];		
			selectMask2[0] = CompareType::EQ;
			selectMask2[1] = CompareType::NC;
			selectMask2[2] = CompareType::NC;
		
			compareMask.push_back(selectMask2);
		
			//set Join mask
			JoinMask innerMask[3];
			innerMask[0] = JoinMask::PRE;
			innerMask[1] = JoinMask::NJ;
			innerMask[2] = JoinMask::NJ;
			
			JoinMask outerMask[3];
			outerMask[0] = JoinMask::PRE;
			outerMask[1] = JoinMask::NJ;
			outerMask[2] = JoinMask::NJ;

			//Creat operation object to pass to query manager
			SelectOperation  selectOp1(&d_queryVector1, selectMask1);
			SelectOperation  selectOp2(&d_queryVector2, selectMask2);
		
			JoinOperation  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), innerMask, outerMask);
		
			std::vector<SelectOperation*> selectOperations;
			std::vector<JoinOperation*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			selectOperations.push_back(&selectOp2);
			joinOperations.push_back(&joinOp);
		
			gettimeofday(&beginEx, NULL);	
			
			queryManager(selectOperations, joinOperations, d_storeVector, fileLength);
			
			//Retrive results from memory
			std::vector<tripleContainer> selectResults = from_mem(*selectOp1.getResult());
			std::vector<tripleContainer> selectResults2 = from_mem(*selectOp2.getResult());
			std::vector<tripleContainer> finalInnerResults = from_mem(*joinOp.getInnerResult());
			std::vector<tripleContainer> finalOuterResults = from_mem(*joinOp.getOuterResult());
			
			cudaDeviceSynchronize();
			gettimeofday(&end, NULL);
		
			float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
			float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			
			timeCuVector.push_back(cuTime);
			timeExVector.push_back(exTime);

			/*
			//Print Results
			cout << "first select result" << endl;
			for (int i = 0; i < selectResults.size(); i++) {
				cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
			}
		
			cout << "second select result" << endl;
			for (int i = 0; i < selectResults2.size(); i++) {
				cout << selectResults2[i].subject << " " << selectResults2[i].predicate << " "  << selectResults2[i].object << endl; 
			}
		
			cout << "final inner result" << endl;
			for (int i = 0; i < finalInnerResults.size(); i++) {
			cout << finalInnerResults[i].subject << " " << finalInnerResults[i].predicate << " "  << finalInnerResults[i].object << endl; 
			} 
			
			cout << "final inner result" << endl;
			for (int i = 0; i < finalOuterResults.size(); i++) {
				cout << finalOuterResults[i].subject << " " << finalOuterResults[i].predicate << " "  << finalOuterResults[i].object << endl; 
			} */
			
			//Print current cycle results
			cout << "first Select Size " << selectResults.size() << endl;
			cout << "second Select Size " << selectResults2.size() << endl;
			cout << "outer  Size " << finalOuterResults.size() << endl;
			cout << "inner Size " << finalInnerResults.size() << endl;			
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;					
			
			cudaFree((*joinOp.getInnerResult()).data());
			cudaFree((*joinOp.getOuterResult()).data());
			cudaFree((*selectOp1.getResult()).data());
			cudaFree((*selectOp2.getResult()).data());
			cudaFree(d_storeVector);
		}
		
		int vecSize = timeCuVector.size();
		float meanCu = 0;
		float meanEx = 0;
		float varianceCu = 0;
		float varianceEx = 0;

		for (int i = 0; i < vecSize; i++) {
			meanCu += timeCuVector[i];
			meanEx += timeExVector[i];
			varianceCu += timeCuVector[i] * timeCuVector[i];
			varianceEx += timeExVector[i] * timeExVector[i];
			cout << timeCuVector[i] << endl;
		}
		meanCu = meanCu / ((float) vecSize);
		varianceCu = varianceCu / ((float) vecSize);
		varianceCu = varianceCu - (meanCu * meanCu);

                meanEx = meanEx / ((float) vecSize);
                varianceEx = varianceEx / ((float) vecSize);
                varianceEx = varianceEx - (meanEx * meanEx);

		cout << "mean cuda time " << meanCu << endl;
		cout << "variance cuda time " << varianceCu << endl;
		
		cout << "mean ex time " << meanEx << endl;
		cout << "variance ex time " << varianceEx << endl;
		return 0;

}



