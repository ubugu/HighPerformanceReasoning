#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>

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
enum compareType {LT, LEQ, EQ, GT, GEQ, NN};

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

		case(compareType::NN):
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
* @param resul : value of the pointer of the query result saved on the gpu device
* @return a vector of int in which are saved the size of each query results.
*/
template<typename element_t>
std::vector<int> rdfSelect(const std::vector<tripleContainer<element_t>*> d_selectQueries, 
	const tripleContainer<element_t>* d_storePointer,
	const int storeSize, 
	std::vector<compareType*> comparatorMask,
	std::vector<tripleContainer<element_t>*>* finalResultPointers) {

		int querySize =  d_selectQueries.size();

		standard_context_t context;		
		auto compact = transform_compact(storeSize, context);
		std::vector<int> resultsSize;
		
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

			tripleContainer<element_t>* d_result;
			cudaMalloc(&d_result, query_count);

			compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
				d_result[dest_index] = d_storePointer[source_index];
			});
			
			finalResultPointers->push_back(d_result);
			resultsSize.push_back(query_count);
		}
		
		return resultsSize;
}


// redefinition of < comparator function.
template<typename element_t>
class TripleComparator
{
	private:
		int joinMaks[3];

        public:
                TripleComparator(int mask[3])
                {
			this.joinMask[0] = mask[0];
			this.joinMask[1] = mask[1];
			this.joinMask[2] = mask[2];
                };

                MGPU_DEVICE bool operator() (tripleContainer<element_t> a, tripleContainer<element_t> b) {
                        if (joinMask[0]) && (a.subject <  b.subject) {
                                return true;
                        }
			
 
                        if (joinMask[1]) && (a.predicate <  b.perdicate) {
                                return true;
                        }

                        if (joinMask[2]) && (a.object <  b.object) {
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
int rdfJoin(tripleContainer<element_t>* innerTable, int innerSize, tripleContainer<element_t>* outerTable, int outerSize, int joinMask[3]) 
{


	//COME PRENDERE I RISULTATI
	standard_context_t context;
	TripleComparator<element_t> comparator = new TripleComparator(joinMask);

	//Sort the two input array
	mergesort<empty_t, tripleContainer<element_t>,  TripleComparator>(innerTable, innerSize , comparator, context);
	mergesort<empty_t, tripleContainer<element_t>,  TripleComparator>(outerTable, outerSize , comparator, context);

	mem_t<int2> joinResult = inner_join<empty_t, tripleContainer<element_t>*,tripleContainer<element_t*, TripleComparator>(innerTable, innserSize, outerTable, outerSize, comparator, context);	

	//Take index results from device memory
	std::vector<int2> result = from_mem(joinResult);
	
	
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

                        h_rdfStore[i] =  {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str())};
                }
                rdfStoreFile.close();

                //take query parameters
                const int TUPLE_LENGHT = 4;
                int queryLenght = (argc - 1) / TUPLE_LENGHT;

                size_t querySize = queryLenght * sizeof(tripleContainer<int>);
                tripleContainer<int>* h_queryVector = (tripleContainer<int>*) malloc(querySize);

                for (int i = 0; i < queryLenght; i++) {
                        int index = 1 + i *  TUPLE_LENGHT;
                        h_queryVector[i] = {atoi(argv[index]), atoi(argv[index + 1]), atoi(argv[index + 2])};
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
		equalMask[1] = equalComp;
		equalMask[2] = equalComp;
		
		compareMask.push_back(equalMask);

                std::vector<tripleContainer<int>*> resultPointer;
		selectResult = rdfSelect<int>(selectQuery, d_storeVector, FILE_LENGHT, compareMask, &resultPointer); 
		
	//	cudaMemcpy(resultsss, selectResult[0] 
		
		
		return 0;
}


