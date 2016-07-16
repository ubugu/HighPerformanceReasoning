
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>
#include <unordered_map>

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


//Redefinition of  comparator function.
template<typename element_t>
class TripleComparator
{
	private:
		int joinMask[3];

        public:
                TripleComparator(int mask[3])
                {
			joinMask[0] = mask[0];
			joinMask[1] = mask[1];
			joinMask[2] = mask[2];
                };

		
                MGPU_DEVICE bool operator() (tripleContainer<element_t> a, tripleContainer<element_t> b) {
                        if ((joinMask[0]) && (a.subject <  b.subject)) {
                                return true;
                        }

                        if ((joinMask[1]) && (a.predicate <  b.predicate)) {
                                return true;
                        }

                        if ((joinMask[2]) && (a.object <  b.object)) {
                                return true;
                        }
			
                        return false;
                };
};


template<typename element_t>
__global__ void indexCopy(tripleContainer<element_t>* src, tripleContainer<element_t>* dest, int2* srcIndex, const bool x) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int currentIndex = 0;
	
	if (x) {
		currentIndex = srcIndex[destIndex].x;
	} else {
		currentIndex = srcIndex[destIndex].y;
	}

	dest[currentIndex] = src[currentIndex];
}


template<typename element_t>
std::vector<mem_t<tripleContainer<element_t>>*> rdfJoin(tripleContainer<element_t>* innerTable, int innerSize, tripleContainer<element_t>* outerTable, int outerSize, int joinMask[3])
{
	standard_context_t context;
	TripleComparator<element_t>* comparator = new TripleComparator<element_t>(joinMask);
	std::vector<mem_t<tripleContainer<element_t>>*> finalResults;

	//Sort the two input array
	mergesort(innerTable, innerSize , *comparator, context);
	mergesort(outerTable, outerSize , *comparator, context);
	
	mem_t<int2> joinResult = inner_join(innerTable, innerSize, outerTable, outerSize, *comparator, context);	


	mem_t<tripleContainer<element_t>>* innerResults = new mem_t<tripleContainer<element_t>>(innerSize, context);
        mem_t<tripleContainer<element_t>>* outerResults = 0;
        
	indexCopy<<<1,joinResult.size()>>>(innerTable, innerResults->data(), joinResult.data(), true);


	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);


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
		int mask[3];

	public:
		JoinOperation(mem_t<tripleContainer <element_t>>** innerTable, mem_t<tripleContainer <element_t>>** outerTable, int mask[3]) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			std::copy(mask, mask + 3, this->mask);
		};
			
		mem_t<tripleContainer <element_t>>** getInnerTable() {
			return this->innerTable;
		};
		
		mem_t<tripleContainer <element_t>>** getOuterTable() {
			return this->outerTable;
		};
		
		int* getMask() {
			return this->mask;
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
		std::vector<mem_t<tripleContainer<element_t>>*>  joinResult = rdfJoin(innerTable->data(), innerTable->size(), outerTable->data(), outerTable->size(), joinOp[i]->getMask());
		joinOp[i]->setResult(joinResult[0]);
	}
	
}

int main(int argc, char** argv) {
 
		using namespace std;
		struct timeval beginPr, beginCu, beginEx, end, beginHash, endHash;


		gettimeofday(&beginPr, NULL);	
		
		using tripleElement = int;
		
		cudaDeviceReset();
		standard_context_t context;
		
                const int FILE_LENGHT = 299829;
              	 
		const int MAX_LENGHT = 100;

                size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer<char[MAX_LENGHT]>);
                tripleContainer<char[MAX_LENGHT]>* h_rdfStore = (tripleContainer<char[MAX_LENGHT]>*) malloc(rdfSize);

                //read store from rdfStore
                ifstream rdfStoreFile ("../rdfStore/rdf2.txt");

                string strInput;

		std::unordered_map<string, int> m;

		
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

                gettimeofday(&beginHash, NULL);
		size = 0;
		for (int i = 0; i < FILE_LENGHT; i++) {
			size = m.size();
			std::pair<std::string,double> subject (h_rdfStore[i].subject, size);
			m.insert(subject);

                        size = m.size();
                        std::pair<std::string,double> predicate (h_rdfStore[i].predicate, size);
                        m.insert(predicate);

                        size = m.size();
                        std::pair<std::string,double> object (h_rdfStore[i].object, size);
                        m.insert(object);


		}		

	        gettimeofday(&endHash, NULL);

                float hashTime = (endHash.tv_sec - beginHash.tv_sec ) * 1000 + ((float) endHash.tv_usec - (float) beginHash.tv_usec) / 1000 ;
		cout << hashTime << endl;

		return 0;
/*
		std::vector<float> timeVector;                

                int N_CYCLE = 100;
		for (int i = 0; i < N_CYCLE; i++) {
			gettimeofday(&beginCu, NULL);


			tripleContainer<tripleElement>* d_storeVector;
			cudaMalloc(&d_storeVector, rdfSize);
			cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
	
			
		        //set Queries (select that will be joined)
		        tripleContainer<tripleElement> h_queryVector1 = {0, 1, 2}; 
		        tripleContainer<tripleElement> h_queryVector2 = {0, 2 , 2};		        
		                     
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
			int joinMask[3];
			joinMask[0] = 1;
			joinMask[1] = 0;
			joinMask[2] = 0;
			
			SelectOperation<tripleElement>  selectOp1(&d_queryVector1, selectMask1);
			SelectOperation<tripleElement>  selectOp2(&d_queryVector2, selectMask2);
		
			JoinOperation<tripleElement>  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), joinMask);
		
			std::vector<SelectOperation<tripleElement>*> selectOperations;
			std::vector<JoinOperation<tripleElement>*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			selectOperations.push_back(&selectOp2);
			joinOperations.push_back(&joinOp);
		
			gettimeofday(&beginEx, NULL);	
			
		
			queryManager<tripleElement>(selectOperations, joinOperations, d_storeVector, FILE_LENGHT);
			
			std::vector<tripleContainer<tripleElement>> selectResults = from_mem(*selectOp1.getResult());
			std::vector<tripleContainer<tripleElement>> selectResults2 = from_mem(*selectOp2.getResult());
			std::vector<tripleContainer<tripleElement>> finalResults = from_mem(*joinOp.getResult());
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
		/*	for (int i = 0; i < selectResults.size(); i++) {
				cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
			}
		
			cout << "second select result" << endl;
	
			cout << selectResults2.size() << endl;
		/*	for (int i = 0; i < selectResults2.size(); i++) {
				cout << selectResults2[i].subject << " " << selectResults2[i].predicate << " "  << selectResults2[i].object << endl; 
			}
		
			cout << "final result" << endl;
			
			cout << finalResults.size() << endl;
			for (int i = 0; i < finalResults.size(); i++) {
				cout << finalResults[i].subject << " " << finalResults[i].predicate << " "  << finalResults[i].object << endl; 
			} 
			
			
			
			cudaFree((*joinOp.getResult()).data());
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
		
		return 0;

	*/
}


