#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>

using namespace mgpu;

//struct to contains a single triple with a element_t type.

const int MAX_LENGHT = 50;

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
/*
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
*/
	return finalResults;
}


/*
* Join enum to define order and which element to join
* NJ indicates a non-join value, so it is ignored during join and sorting
* So that it improves performance avoiding uneecessary conditional expression
*/
enum class JoinMask {NJ = -1, SBJ = 0, PRE = 1, OBJ = 2};


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
	
	
}

int main(int argc, char** argv) {
 
		using namespace std;
		struct timeval beginPr, beginCu, beginEx, end;
		gettimeofday(&beginPr, NULL);	
		cudaDeviceReset();
		standard_context_t context;
		
		ifstream rdfStoreFile ("../rdfStore/rdfTimeStr.txt");
		string strInput;

		//Settare lenght variabile
       		const int FILE_LENGHT = 302924;
              	           
                size_t rdfSize = FILE_LENGHT  * sizeof(tripleContainer);
                tripleContainer* h_rdfStore = (tripleContainer*) malloc(rdfSize);

		int size = 0;		
		char emptyBuff[MAX_LENGHT] = {0};
	
                for (int i = 0; i < FILE_LENGHT; i++) {
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
/*
		std::vector<float> timeVector;                

                int N_CYCLE = 10;
		for (int i = 0; i < N_CYCLE; i++) {

                    //    string  current = "<http://example.org/int/" + to_string(i + 100) + ">";
			auto str = "<http://example.org/int/1>";
			auto str2 = "<http://example.org/int/0>";
                  //      const char* str = current.c_str();
                       
		//	string cicle = "<http://example.org/intt/" + to_string(i) + ">";
                       // const char* str2 = cicle.c_str();


			gettimeofday(&beginCu, NULL);

			tripleContainer* d_storeVector;
			cudaMalloc(&d_storeVector, rdfSize);
			cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
	
			//Use query "SELECT * WHERE {  ?s ?p  <http://example.org/int/1>.  <http://example.org/int/0> ?p  ?o} ";
			
		        //set Queries (select that will be joined)
		        tripleContainer h_queryVector1; 
		        tripleContainer h_queryVector2;
		        
		        char object1[MAX_LENGHT] = {0};  
			
			std::copy(str, str + 26, object1);
		        
		     	strncpy(h_queryVector1.subject, emptyBuff, MAX_LENGHT);      	
		     	strncpy(h_queryVector1.predicate, emptyBuff, MAX_LENGHT);
		     	strncpy(h_queryVector1.object, object1, MAX_LENGHT);
		     	
		     	char subject2[MAX_LENGHT] = {0};  
		     
			std::copy(str2, str2 + 26, subject2);
	
			cout << "strings are " << str << " " << str2 << endl;
		     	
		        strncpy(h_queryVector2.subject, subject2, MAX_LENGHT);      	
		     	strncpy(h_queryVector2.predicate, emptyBuff, MAX_LENGHT);
		     	strncpy(h_queryVector2.object, emptyBuff, MAX_LENGHT);
		 	         	              
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
			
		//	queryManager(selectOperations, joinOperations, d_storeVector, FILE_LENGHT);
			
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
			
			timeVector.push_back(cuTime);
			
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
			} 
			
			//Print current cycle results
			cout << "first Select Size " << selectResults.size() << endl;
			cout << "first Select Size " << selectResults2.size() << endl;
			cout << "first Select Size " << finalOuterResults.size() << endl;
			cout << "first Select Size " << finalInnerResults.size() << endl;			
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;					
			
			cudaFree((*joinOp.getInnerResult()).data());
			cudaFree((*joinOp.getOuterResult()).data());
			cudaFree((*selectOp1.getResult()).data());
			cudaFree((*selectOp2.getResult()).data());
			cudaFree(d_storeVector);
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
*/		
		return 0;

}



