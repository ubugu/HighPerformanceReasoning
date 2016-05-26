
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>

//Define callable function from host and deivce 
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


using namespace mgpu;

//struct to contains a single triple.
struct tripleContainer {
	int subject;
	int predicate;
	int object;
	int padding_4;
};

struct deviceQueryResult {
	mem_t<tripleContainer>* dataPointer;
	int dataSize;
};


// redefinition of < comparator function.
class TripleComparator
{
        public:
                CUDA_CALLABLE_MEMBER TripleComparator()
                {
                };

                CUDA_CALLABLE_MEMBER bool operator() (tripleContainer a, tripleContainer b) {
                        if (a.subject <  b.subject) {
                                return true;
                        }

/*                    
                        if ((a.subject == b.subject) && (a.predicate < b.predicate)) {
                                return true;
                        }

                        if ((a.subject == b.subject) && (a.predicate == b.predicate) && (a.object < b.object)) {
                                return true;
                        }
*/
                        return false;
                };
};

CUDA_CALLABLE_MEMBER bool operator <(const tripleContainer x, const tripleContainer y) {
        TripleComparator compare;
        return compare(x,y);
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


int main(int argc, char** argv) {
	{
		using namespace std;

		//Check the number of argument passed
		if ((argc - 1) %  4 != 0) {
			cout << "wrong number of input elements" << endl;
			return -1;
		}
		
                const int FILE_LENGHT = 100000;
		size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer);
		
		tripleContainer* h_rdfStore = (tripleContainer*) malloc(rdfSize);

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

                size_t querySize = queryLenght * sizeof(tripleContainer);
                tripleContainer* h_queryVector = (tripleContainer*) malloc(querySize);
		
		for (int i = 0; i < queryLenght; i++) {
			int index = 1 + i *  TUPLE_LENGHT;
	                h_queryVector[i] = {atoi(argv[index]), atoi(argv[index + 1]), atoi(argv[index + 2])};
			cout << atoi(argv[index]) << endl;
		}
                

		standard_context_t context;
		mem_t<tripleContainer> d_rdfStore(FILE_LENGHT,context); 
		cudaMemcpy(d_rdfStore.data(), h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	

		mem_t<tripleContainer> d_queryVector(queryLenght, context);
                


		auto compact = transform_compact(FILE_LENGHT, context);
		tripleContainer* rdfStorePointer = d_rdfStore.data();
		tripleContainer* queryVectorPointer = d_queryVector.data();

		std::vector<vector<tripleContainer>> finalResult;
		std::vector<deviceQueryResult> h_queryResults;

		for (int i = 0; i < queryLenght; i++) {
	                
	                cudaMemcpy(d_queryVector.data(), &h_queryVector[i], sizeof(tripleContainer), cudaMemcpyHostToDevice);

			int query_count = compact.upsweep([=] MGPU_DEVICE(int index) {
				bool subjectEqual = false;
				bool predicateEqual = false;
				bool objectEqual = false;

				if (queryVectorPointer->subject != -1) {
					subjectEqual = (queryVectorPointer->subject == rdfStorePointer[index].subject);
				} else {
					subjectEqual = true;
				}

				if (queryVectorPointer->predicate != -1) {
					predicateEqual = (queryVectorPointer->predicate == rdfStorePointer[index].predicate);
				} else {
					predicateEqual = true;
				}
	
				if (queryVectorPointer->object != -1) {
	                               	objectEqual = (queryVectorPointer->object == rdfStorePointer[index].object);
                  	      } else {
                               		objectEqual = true;
                        	}

				return (subjectEqual && predicateEqual && objectEqual);
			});


			mem_t <tripleContainer> d_queryResult(query_count, context);
			deviceQueryResult newResult = {&d_queryResult, query_count};			

			h_queryResults.push_back(newResult);

			tripleContainer* queryResultPointer = d_queryResult.data();
		
			compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
				queryResultPointer[dest_index] = rdfStorePointer[source_index];
			});

			cout << "the result will be " << endl;
                        vector<tripleContainer> results = from_mem(*h_queryResults[i].dataPointer);
                        for (int k = 0; k < results.size(); k++) {
                                cout << results[k].subject << " " << results[k].predicate << " " << results[k].object << endl;
                        }

		
			std::vector<tripleContainer> h_queryResult = from_mem(d_queryResult);
			finalResult.push_back(h_queryResult);
		}
		
		for (int i =0; i < finalResult.size(); i++) {
			cout << "Result for the " << i << " query:" << endl;
			for (int k =0; k < finalResult[i].size() ; k++) {
				cout << finalResult[i][k].subject << " " << finalResult[i][k].predicate << " " << finalResult[i][k].object << endl;
			}
		}
			
		cout << "result test" << endl;
		for (int i =0; i < 2; i ++ ){
			cout << i << " results" << endl;
			vector<tripleContainer> results = from_mem(*h_queryResults[i].dataPointer);
			for (int k = 0; k < results.size(); k++) {			
				cout << results[k].subject << " " << results[k].predicate << " " << results[k].object << endl;
			}
		}

/*
		if (finalResult.size() == 2) {
			cout << "starting join on the subject" << endl;
			TripleComparator comparator;
			mem_t<int2> joinResult = inner_join<empty_t, tripleContainer*, tripleContainer*, TripleComparator >(h_queryResults[0].dataPointer->data(), h_queryResults[0].dataSize, h_queryResults[1].dataPointer->data(), h_queryResults[1].dataSize, comparator, context);
			
			
		}

*/		
	}
	
}


