

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

//struct to contains a single triple.
struct tripleContainer {
	int subject;
	int predicate;
	int object;
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


int main(int argc, char** argv) {
	{
		using namespace std;

		//Check the number of argument passed
		if (argc < 5 ) {
			cout << "argument mismatch ";
			return -1;
		}
		
                const int FILE_LENGHT = 100000;
		size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer);
		
		tripleContainer* h_rdfStore = (tripleContainer*) malloc(rdfSize);

		//read store from rdfStore
		ifstream rdfStoreFile (argv[1]);
		if (rdfStoreFile.fail()) {
			cout << "file not found" << endl;
			return -1;
		}

		string strInput;

		for (int i = 0; i < FILE_LENGHT; i++) {
			getline(rdfStoreFile,strInput);
									
			std::vector<string> triple ;
			separateWords(strInput, triple, ' ');
						
			h_rdfStore[i] =  {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str())};
		}		
		rdfStoreFile.close();


                //take query parameters
                int queryLenght = 1;
		
                size_t querySize = queryLenght * sizeof(tripleContainer);
                tripleContainer* h_queryVector = (tripleContainer*) malloc(querySize);

	        h_queryVector[0] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4])};
		
         
		standard_context_t context;
		mem_t<tripleContainer> d_rdfStore(FILE_LENGHT,context); 
		cudaMemcpy(d_rdfStore.data(), h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	

		mem_t<tripleContainer> d_queryVector(queryLenght, context);
                


		auto compact = transform_compact(FILE_LENGHT, context);
		tripleContainer* rdfStorePointer = d_rdfStore.data();
		tripleContainer* queryVectorPointer = d_queryVector.data();

		

		
	                
                cudaMemcpy(d_queryVector.data(), h_queryVector, sizeof(tripleContainer), cudaMemcpyHostToDevice);

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
		tripleContainer* queryResultPointer = d_queryResult.data();

		compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
			queryResultPointer[dest_index] = rdfStorePointer[source_index];
		});

		
		std::vector<tripleContainer> h_queryResult = from_mem(d_queryResult);

		
		for (int k =0; k < query_count ; k++) {
			cout << h_queryResult[k].subject << " " << h_queryResult[k].predicate << " " << h_queryResult[k].object << endl;
		}
		
	}
	
}


