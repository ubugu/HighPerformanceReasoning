
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>

//Define callable function from host and deivce 
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

using namespace mgpu;

//struct to contains a single triple with a element_t type.
template<typename element_t>
struct tripleContainer {
        element_t subject;
        element_t predicate;
        element_t object;
};

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

//Make a select query on the store pointer saved on the
template<typename element_t>
std::vector<tripleContainer<element_t>*> ssselect(const std::vector<tripleContainer<element_t>*> d_selectQueries, const tripleContainer<element_t>* d_storePointer, const int storeSize, const element_t variableToken) {

		int querySize =  d_selectQueries.size();
		
		standard_context_t context;		
		auto compact = transform_compact(storeSize, context);
		std::vector<tripleContainer<element_t>*> finalResultPointers;

		for (int i = 0; i < querySize; i++) {
			tripleContainer<element_t>* currentPointer = d_selectQueries[i]; 
	             
			int query_count = compact.upsweep([=] MGPU_DEVICE(int index) {
				bool subjectEqual = false;
				bool predicateEqual = false;
				bool objectEqual = false;

				if (currentPointer->subject != variableToken) {
					subjectEqual = (currentPointer->subject == d_storePointer[index].subject);
				} else {
					subjectEqual = true;
				}

				if (currentPointer->predicate != variableToken) {
				predicateEqual = (currentPointer->predicate == d_storePointer[index].predicate);
				} else {
				predicateEqual = true;
				}
	
				if (currentPointer->object != variableToken) {
					objectEqual = (currentPointer->object == d_storePointer[index].object);
				} else {
					objectEqual = true;
				}

				return (subjectEqual && predicateEqual && objectEqual);
			});

			tripleContainer<element_t>* d_result;
			cudaMalloc(&d_result, query_count);

			compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
				d_result[dest_index] = d_storePointer[source_index];
			});
			
			finalResultPointers.push_back(d_result);
		}
		
		return finalResultPointers;
		
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
		std::vector<tripleContainer<int>*> selectResult;
		selectQuery.push_back(d_queryVector);
		int token = -1;
		selectResult = ssselect<int>(selectQuery, d_storeVector, FILE_LENGHT, token); 

		return 0;
}


