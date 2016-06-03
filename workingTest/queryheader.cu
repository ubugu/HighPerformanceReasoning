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
	element_t padding_4;
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
		case(compareType::LT):
			return a < b;

		case(compareType::LEQ):
			return a <= b;

		case(compareType::EQ):	
			return a == b;

		case(compareType::GEQ):
			return a >= b;

		case(compareType::GT):
			return a > b;

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
	int querySize =  d_selectQueries.size();

	standard_context_t context; 		
	auto compact = transform_compact(storeSize, context);
	std::vector<mem_t<tripleContainer<element_t>>*> finalResults;

	for (int i = 0; i < querySize; i++) {
		tripleContainer<element_t>* currentPointer = d_selectQueries[i]; 
			
		compareType subjectComparator = comparatorMask[i][0];	             	
		compareType predicateComparator = comparatorMask[i][1];
		compareType objectComparator = comparatorMask[i][2];

		int query_count = compact.upsweep([=] MGPU_DEVICE(int index) {
			bool subjectEqual = false;
			bool predicateEqual = false;
			bool objectEqual = false;

			subjectEqual = compare<element_t>(d_storePointer[index].subject, currentPointer->subject, subjectComparator);
			predicateEqual = compare<element_t>(d_storePointer[index].predicate, currentPointer->predicate, predicateComparator);
			objectEqual = compare<element_t>(d_storePointer[index].object, currentPointer->object, objectComparator);

			return (subjectEqual && predicateEqual && objectEqual);
		});

		mem_t<tripleContainer<element_t>>* currentResult = new mem_t<tripleContainer<element_t>>(query_count, context);
		tripleContainer<element_t>* d_currentResult =  currentResult->data();

		compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
			d_currentResult[dest_index] = d_storePointer[source_index];
		});
		
		finalResults.push_back(currentResult);
	}
	return finalResults;
}


// redefinition of < comparator function.
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
			
 
                        if ((joinMask[1]) && (a.predicate <  b.perdicate)) {
                                return true;
                        }

                        if ((joinMask[2]) && (a.object <  b.object)) {
                                return true;
                        }
			
                        return false;
                };
};

/*
MGPU_DEVICE bool operator <(const tripleContainer x, const tripleContainer y) {
        TripleComparator compare;
        return compare(x,y);
}*/


template<typename element_t>
MGPU_DEVICE void indexCopy(tripleContainer<element_t>* src, tripleContainer<element_t>* dest, int2* srcIndex, const bool x) 
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

template<typename element_t>
std::vector<tripleContainer<element_t>*> rdfJoin(tripleContainer<element_t>* innerTable, int innerSize, tripleContainer<element_t>* outerTable, int outerSize, int joinMask[3]) 
{


	//COME PRENDERE I RISULTATI
	standard_context_t context;
	TripleComparator<element_t> comparator = new TripleComparator<element_t>(joinMask);

	//Sort the two input array
	mergesort<empty_t, tripleContainer<element_t>, TripleComparator>(innerTable, innerSize , comparator, context);
	mergesort<empty_t, tripleContainer<element_t>, TripleComparator>(outerTable, outerSize , comparator, context);

	mem_t<int2> joinResult = inner_join<empty_t, tripleContainer<element_t>*,tripleContainer<element_t>*, TripleComparator>(innerTable, innerSize, outerTable, outerSize, comparator, context);	

	//Take index results from device memory
	tripleContainer<element_t>* innerResults;
	tripleContainer<element_t>* outerResults;

	//TODO SETTARE VALORE SIZE
	//size_t resultSize = 0 * sizeof(tripleContainer<element_t>);
	size_t resultSize = joinResult._size;
	cudaMalloc(&innerResults, resultSize);
	cudaMalloc(&outerResults, resultSize);
		
	indexCopy(innerTable, innerResults, joinResult.data(), true);
	indexCopy(outerTable, outerResults, joinResult.data(), false);

	std::vector<tripleContainer<element_t>*> finalResult;
	finalResult.push_back(innerResults);
	finalResult.push_back(outerResults);
	return finalResult;
}



int main(int argc, char** argv) {
		using namespace std;
		if (argc < 4 ) {
			std::cout << "errore " << endl;
		}

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

                        h_rdfStore[i] =  {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str()), 0};
                }
                rdfStoreFile.close();

                //take query parameters
                const int TUPLE_LENGHT = 4;
                int queryLenght = (argc - 1) / TUPLE_LENGHT;

                size_t querySize = queryLenght * sizeof(tripleContainer<int>);
                tripleContainer<int>* h_queryVector = (tripleContainer<int>*) malloc(querySize);

                for (int i = 0; i < queryLenght; i++) {
                        int index = 1 + i *  TUPLE_LENGHT;
                        h_queryVector[i] = {atoi(argv[index]), atoi(argv[index + 1]), atoi(argv[index + 2]), 0};
                }

		tripleContainer<int>* d_storeVector;
		cudaMalloc(&d_storeVector, rdfSize);
		cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);		
		tripleContainer<int>* d_queryVector;
		cudaMalloc(&d_queryVector, sizeof(tripleContainer<int>));
		cudaMemcpy(d_queryVector, h_queryVector, sizeof(tripleContainer<int>), cudaMemcpyHostToDevice);
		
		std::vector<tripleContainer<int>*> selectQuery;
		std::vector<int> selectResult;
		selectQuery.push_back(d_queryVector);
		compareType equalComp = compareType::EQ;
		std::vector<compareType*> compareMask;
		compareType equalMask[3];
		
		equalMask[0] = equalComp;
		equalMask[1] = compareType::NC;
		equalMask[2] = compareType::NC;
		
		compareMask.push_back(equalMask);

                std::vector<tripleContainer<int>*> resultPointer;


		std::vector<mem_t<tripleContainer<int>>*> selectR = rdfSelect<int>(selectQuery, d_storeVector, FILE_LENGHT, compareMask); 
			
		std::vector<tripleContainer<int>> provasd = from_mem(*selectR[0]);

		for (int i = 0; i < provasd.size(); i++ ) {
			std::cout << provasd[i].subject << " " << provasd[i].predicate << " " << provasd[i].object << std::endl;
		}

		return 0;
}


