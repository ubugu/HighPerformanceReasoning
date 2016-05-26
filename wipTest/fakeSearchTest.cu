#include <iostream>
#include <fstream>
#include <cstdlib>
#include<moderngpu/kernel_sortedsearch.hxx>
//#include <moderngpu/kernel_mergesort.hxx>

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
	int object;
	int predicate;
	int padding_4;
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
			
			if ((a.subject == b.subject) && (a.predicate == -1)) {
                                return false;
                        }

                        if ((a.subject == b.subject) && (a.predicate == b.predicate) && (a.object == -1)) {
                                return false;
                        }


                        if ((a.subject == b.subject) && (a.predicate < b.predicate)) {
                                return true;
                        }
			
                        if ((a.subject == b.subject) && (a.predicate == b.predicate) && (a.object < b.object)) {
                                return true;
                        }

                        return false;
                }
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

		if (argc < 4)
			return -1;
		
                const int FILE_LENGHT = 100000;
		size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer);
		tripleContainer* memoryStore = (tripleContainer*) malloc(rdfSize);

	
		//read store from file
//		ifstream rdfStore ("rdfTest.txt");
		ifstream rdfStore ("rdfSorted.txt");

			
		string strInput;

		for (int i = 0; i < FILE_LENGHT; i++) {
			getline(rdfStore,strInput);
									
			std::vector<string> triple ;
			separateWords(strInput, triple, ' ');
						
			memoryStore[i] =  {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str())};
		}		
		rdfStore.close();

		tripleContainer* rdfPointer_d;
		cudaMalloc(&rdfPointer_d, rdfSize);
		cudaMemcpy(rdfPointer_d, memoryStore, rdfSize, cudaMemcpyHostToDevice);
		
		TripleComparator comparator;
/*                
		cout << "start Sorting" << endl;
		standard_context_t context;
		mergesort<empty_t, tripleContainer,  TripleComparator>(rdfPointer_d, FILE_LENGHT , comparator, context);	  
*/



		//calculate lower and upper bounds
		int queryLenght = 1;
                size_t querySize = queryLenght * sizeof(tripleContainer);
		tripleContainer* queryVector = (tripleContainer*) malloc(querySize);
		queryVector[0] = {atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
		
		tripleContainer *queryVector_d;
		cudaMalloc(&queryVector_d, querySize);
		cudaMemcpy(queryVector_d, queryVector, querySize, cudaMemcpyHostToDevice);
		
		int *queryResult_d;
		size_t resultSize = (queryLenght)  * sizeof(int);
		cudaMalloc(&queryResult_d, resultSize);
		TripleComparator secondComparator;

          	standard_context_t context;

		cout << "computing lower bounds " << endl;		
	        sorted_search<bounds_lower,empty_t,tripleContainer*, tripleContainer*, int*, TripleComparator>(queryVector_d, queryLenght, rdfPointer_d, FILE_LENGHT, queryResult_d, comparator, context);

		int* lowerBound = (int*) malloc(queryLenght * sizeof(int)) ;
		cout << "ended " << endl;
		cudaMemcpy(lowerBound, queryResult_d, resultSize, cudaMemcpyDeviceToHost);	
	
                cout << "computing upper bounds " << endl;
                sorted_search<bounds_upper,empty_t,tripleContainer*, tripleContainer*, int*, TripleComparator>(queryVector_d, queryLenght, rdfPointer_d, FILE_LENGHT, queryResult_d, secondComparator, context);
                int* upperBound = (int*) malloc(queryLenght * sizeof(int)) ;
                cudaMemcpy(upperBound, queryResult_d, resultSize, cudaMemcpyDeviceToHost);


		cudaFree(queryResult_d);
		cudaFree(queryVector_d); 
		cudaFree(rdfPointer_d);

		for (int i =0; i < queryLenght; i++) {
			cout << "upper del " << i << " valore " <<  upperBound[i] << endl;
			cout << "upper del " << i << " valore " <<  lowerBound[i] << endl;
		}

	}
	
}


