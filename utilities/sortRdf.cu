#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include<moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_mergesort.hxx>

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

CUDA_CALLABLE_MEMBER bool operator <(const tripleContainer& x, const tripleContainer& y) {
        TripleComparator compare;
//	printf("let me compare pls ");

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
				
		//load store in memory
		vector<tripleContainer> memoryStore;

		

		//read store from file
		
		fstream rdfStore (argv[1]);
			
		string strInput;
		const int FILE_LENGHT = 10000;

		for (int i = 0; i < FILE_LENGHT; i++) {
			getline(rdfStore,strInput);
			
			cout << i << endl;			
			std::vector<string> triple ;
			separateWords(strInput, triple, ' ');
						
			tripleContainer newTriple {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str())};
			
			memoryStore.push_back(newTriple);
		}

		
		cout << "finito " << endl;

		tripleContainer *rdfPointer_h = &memoryStore[0];
		size_t size = memoryStore.size() * sizeof(tripleContainer);

		tripleContainer *rdfPointer_d;
		cudaMalloc(&rdfPointer_d, size);

		cudaMemcpy(rdfPointer_d, rdfPointer_h, size, cudaMemcpyHostToDevice);
                

		cout << "start Sorting" << endl;
		standard_context_t context;
		TripleComparator comparator;
		mergesort<empty_t, tripleContainer,  TripleComparator>(rdfPointer_d, FILE_LENGHT , comparator, context);	  
		cudaMemcpy(rdfPointer_h, rdfPointer_d, size, cudaMemcpyDeviceToHost);
		cudaFree(rdfPointer_d);

		cout << " ho sortato" << endl;

		for (int i =0; i < 30; i ++) {
			tripleContainer triple = rdfPointer_h[i];
			                        cout << triple.subject << " " << triple.predicate << " " << triple.object << " ." << endl;
		}
	
		ofstream sorted("rdfSorted.txt");
		for (int i =0; i <FILE_LENGHT; i++) {
			tripleContainer triple = rdfPointer_h[i];
			sorted << triple.subject << " " << triple.predicate << " " << triple.object << " ." << endl;
			
		}
		sorted.close();
		rdfStore.close();
	}
	
}


