
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


// redefinition of < comparator function.
class TripleComparator
{
        public:
                CUDA_CALLABLE_MEMBER TripleComparator()
                {
                };

                CUDA_CALLABLE_MEMBER bool operator() (int a, int b) {
                        if (a <  b) {
                                return true;
                        }

                        return false;
                };
};




int main(int argc, char** argv) {
	{
		using namespace std;

				
                const int INNER_SIZE = 4;
		const int OUTER_SIZE = 4;

		size_t innerSize = INNER_SIZE * sizeof(int);
		size_t outerSize = OUTER_SIZE * sizeof(int);
		
		int* innerTable = (int*) malloc(innerSize);
		int* outerTable = (int*) malloc(outerSize);
		
 
		//Inizialize inner table
		for (int i = 0; i <INNER_SIZE; i++ ) {
			innerTable[i] = i;
		}


                //Inizialize outer table
                for (int i = 0; i <OUTER_SIZE; i++ ) {
                        outerTable[i] = 2* i;
                }
               
		standard_context_t context;

		mem_t<int> d_innerTable(INNER_SIZE, context);
		mem_t<int> d_outerTable(OUTER_SIZE, context);
		cudaMemcpy(d_innerTable.data(), innerTable, innerSize, cudaMemcpyHostToDevice);
                cudaMemcpy(d_outerTable.data(), outerTable, outerSize, cudaMemcpyHostToDevice);

	
		cout << "starting join on the subject" << endl;
		TripleComparator comparator;
		mem_t<int2> joinResult = inner_join<empty_t, int*, int*, TripleComparator >(d_innerTable.data(), INNER_SIZE, d_outerTable.data(), OUTER_SIZE, comparator, context);
			
			
		std::vector<int2> result = from_mem(joinResult);		
		for (int i =0; i < result.size(); i++){
			std::cout << result[i].x << " " << result[i].y << std::endl;
		}
	}
	
}


