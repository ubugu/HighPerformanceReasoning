#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>

using namespace mgpu;

//struct to contains a single triple with a element_t type.
template<typename element_t>
struct tripleContainer {
        element_t subject;
        element_t predicate;
        element_t object;
	element_t padding;
};

/**
* Enum for condition that are applied 
* to the triple, and function associated
* to them.
**/
enum compareType {LT, LEQ, EQ, GT, GEQ, NC};

template<typename element_t>
MGPU_DEVICE bool compare(element_t a, element_t b, compareType type) {
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

			subjectEqual = compare<element_t>(d_storePointer[index].subject, currentPointer->subject, subjectComparator);
			predicateEqual = compare<element_t>(d_storePointer[index].predicate, currentPointer->predicate, predicateComparator);
			objectEqual = compare<element_t>(d_storePointer[index].object, currentPointer->object, objectComparator);

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

//Functions for copying the index values of join into array
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

	dest[destIndex] = src[currentIndex];
}


/***
** RDF join
***/
template<typename element_t>
std::vector<mem_t<tripleContainer<element_t>>*> rdfJoin(tripleContainer<element_t>* innerTable, int innerSize, tripleContainer<element_t>* outerTable, int outerSize, int joinMask[3])
{
	standard_context_t context;
	TripleComparator<element_t>* comparator = new TripleComparator<element_t>(joinMask);

	//Sort the two input array
	mergesort<empty_t, tripleContainer<element_t>, TripleComparator<element_t>>(innerTable, innerSize , *comparator, context);
	mergesort<empty_t, tripleContainer<element_t>, TripleComparator<element_t>>(outerTable, outerSize , *comparator, context);

	mem_t<int2> joinResult = inner_join<empty_t, tripleContainer<element_t>*,tripleContainer<element_t>*, TripleComparator<element_t>>(innerTable, innerSize, outerTable, outerSize, *comparator, context);	

	std::vector<mem_t<tripleContainer<element_t>>*> finalResults;

	mem_t<tripleContainer<element_t>>* innerResults = new mem_t<tripleContainer<element_t>>(joinResult.size(), context);
        mem_t<tripleContainer<element_t>>* outerResults = new mem_t<tripleContainer<element_t>>(joinResult.size(), context);

	indexCopy<<<1,joinResult.size()>>>(innerTable, innerResults->data(), joinResult.data(), true);
	indexCopy<<<1,joinResult.size()>>>(outerTable, outerResults->data(), joinResult.data(), false);

	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);


	return finalResults;
}


//Section for defining operation classes
template <typename element_t>
class RelationalOperation
{
	typedef mem_t<tripleContainer <element_t>> opElem_t;

	private:
		opElem_t* result = 0;	
	public:
		opElem_t* getResult() {
			return this->result;
		};
		
		void setResult(opElem_t* result) {
			this->result = result;
		};
		
};

template <typename element_t>
class JoinOperation : public RelationalOperation<element_t>
{
	typedef mem_t<tripleContainer <element_t>> opElem_t;
	
	private:
		opElem_t* innerTable;
		opElem_t* outerTable;
		int mask[3];

	public:
		JoinOperation(opElem_t* innerTable, opElem_t* outerTable, int mask[3]) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			std::copy(mask, mask + 3, this->mask);
		};
			
		opElem_t* getInnerTable() {
			return this->innerTable;
		};
		
		opElem_t* getOuterTable() {
			return this->outerTable;
		};
		
		int* getMask() {
			return this->mask;
		};		
};

template<typename element_t>
class SelectOperation : public RelationalOperation<element_t>
{
	typedef mem_t<tripleContainer <element_t>> opElem_t;
	
	private:
		opElem_t* query;
		compareType operationMask[3];

	public:
		SelectOperation(opElem_t* query, compareType operationMask[3]) {
			this->query = query;
			std::copy(operationMask, operationMask + 3, this->operationMaskoperationMask);
		};
			
		opElem_t* getQuery() {
			return this->query;
		};
		
		compareType* getOperationMask() {
			return this->operationMask;
		};
};


template <typename element_t>
void queryManager(std::vector<SelectOperation<element_t>> selectOp, std::vector<JoinOperation<element_t>> joinOp, const tripleContainer<element_t>* d_storePointer, const int storeSize) {

	std::vector<tripleContainer<element_t>*> d_selectQueries;
	std::vector<compareType*> comparatorMask;
	
	for (int i = 0; i < selectOp.size(); i++) {
		d_selectQueries.push_back(selectOp[i].getQuery().data());
		comparatorMask.push_back(selectOp[1].getOperationMask());
	}

	std::vector<mem_t<tripleContainer<element_t>>*> selectResults = rdfSelect(d_selectQueries, d_storePointer, storeSize, comparatorMask);
	
	for (int i = 0; i < selectResults.size(); i++) {
		for (int k = 0; k < selectOp.size(); k++) {
			selectOp[k].setResult(selectResults[i]);
		}
	}
	
	
	/***
	** TODO : FOR NOW KEEP THE INNER TABLE RESULT AS THE RESULT FOR EACH JOIN. cHECK IF THIS IS CORRECT;
	** THE PARSER WILL TAKE HANDLE OF INSERTING IN THE INNER TABLE THE RIGHT ELEMENT
	***/
	for (int i = 0; i < joinOp; i++) {
		std::vector<mem_t<tripleContainer<element_t>>*>  joinResult = rdfJoin(joinOp[i].getInnerTable.data(), joinOp[i].getInnerTable.size(), joinOp[i].getOuterTable.data(), joinOp[i].getOuterTable.size(), *joinOp[i].getMask());	
		
		joinOp.setResult(joinResult[0]);
	}
	
	
}



int main(int argc, char** argv) {
		using namespace std;


                const int FILE_LENGHT = 100000;
                size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer<int>);
                tripleContainer<int>* h_rdfStore = (tripleContainer<int>*) malloc(rdfSize);

                //read store from rdfStore
                ifstream rdfStoreFile ("../rdfStore/rdfSorted.txt");

                string strInput;

                for (int i = 0; i < FILE_LENGHT; i++) {
                        getline(rdfStoreFile,strInput);

                        std::vector<string> triple ;
                        separateWords(strInput, triple, ' ');

			h_rdfStore[i].subject = atoi(triple[0].c_str());
			h_rdfStore[i].predicate = atoi(triple[1].c_str());
			h_rdfStore[i].object = atoi(triple[2].c_str());
                }
                rdfStoreFile.close();

                //take query parameters
                const int TUPLE_LENGHT = 4;
                int queryLenght = (argc - 1) / TUPLE_LENGHT;

		//crete select query
                size_t querySize = queryLenght * sizeof(tripleContainer<int>);
                tripleContainer<int>* h_queryVector = (tripleContainer<int>*) malloc(querySize);

		h_queryVector[0].subject = 0;
		h_queryVector[0].predicate = 200;
		h_queryVector[0].object = -1;

		//store values on device
		tripleContainer<int>* d_storeVector;
		cudaMalloc(&d_storeVector, rdfSize);
		cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
			
		tripleContainer<int>* d_queryVector;
		cudaMalloc(&d_queryVector, sizeof(tripleContainer<int>));
		cudaMemcpy(d_queryVector, h_queryVector, sizeof(tripleContainer<int>), cudaMemcpyHostToDevice);
		
		tripleContainer<int>* d_queryVector2;
		cudaMalloc(&d_queryVector2, sizeof(tripleContainer<int>));
		cudaMemcpy(d_queryVector2, h_queryVector, sizeof(tripleContainer<int>), cudaMemcpyHostToDevice);
		
	
		std::vector<tripleContainer<int>*> selectQuery;
		selectQuery.push_back(d_queryVector);
		selectQuery.push_back(d_queryVector2);

		//Create mask operation for select
		std::vector<compareType*> compareMask;
		compareType equalMask[3];
		
		equalMask[0] = compareType::EQ;
		equalMask[1] = compareType::LT;
		equalMask[2] = compareType::NC;

		compareMask.push_back(equalMask);
		
		compareType equalMask2[3];		
		equalMask2[0] = compareType::EQ; 
		equalMask2[1] = compareType::GT;
		equalMask2[2] = compareType::NC;
		
		compareMask.push_back(equalMask2);
		
		std::vector<mem_t<tripleContainer<int>>*> selectR = rdfSelect<int>(selectQuery, d_storeVector, FILE_LENGHT, compareMask); 
			
		tripleContainer<int>* innerTable = selectR[0]->data();
		tripleContainer<int>* outerTable = selectR[1]->data();
		
		int joinMask[3];
		joinMask[0] = 1;
		joinMask[1] = 0;
		joinMask[2] = 0;
				
		vector<mem_t<tripleContainer<int>>*> joinR;		

		joinR = rdfJoin(innerTable, selectR[0]->size(), outerTable, selectR[1]->size(), joinMask);
		std::vector<tripleContainer<int>> h_result = from_mem(*selectR[0]);
		
		cout << "first select " << endl;
		for (int i = 0; i < h_result.size(); i++ ) {
			std::cout << h_result[i].subject << " " << h_result[i].predicate << " " << h_result[i].object << std::endl;
		}

		cout << "second select " << endl;
		h_result = from_mem(*selectR[1]);
                for (int i = 0; i < h_result.size(); i++ ) {
                        std::cout << h_result[i].subject << " " << h_result[i].predicate << " " << h_result[i].object << std::endl;
                }
		
		cout << "first join " << endl;
                h_result = from_mem(*joinR[0]);
                for (int i = 0; i < h_result.size(); i++ ) {
                        std::cout << h_result[i].subject << " " << h_result[i].predicate << " " << h_result[i].object << std::endl;
                }

		cout << "second join " << endl;
                h_result = from_mem(*joinR[1]);
                for (int i = 0; i < h_result.size(); i++ ) {
                        std::cout << h_result[i].subject << " " << h_result[i].predicate << " " << h_result[i].object << std::endl;
                }


		return 0;

}
