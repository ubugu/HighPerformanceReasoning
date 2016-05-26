
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

                CUDA_CALLABLE_MEMBER bool operator() (tripleContainer a, tripleContainer b) {
                        if (a.subject <  b.subject) {
                                return true;
                        }
 
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

				
                const int INNER_SIZE = 5;
		const int OUTER_SIZE = 5;

		size_t innerSize = INNER_SIZE * sizeof(tripleContainer);
		size_t outerSize = OUTER_SIZE * sizeof(tripleContainer);
		
		tripleContainer* innerTable = (tripleContainer*) malloc(innerSize);
		tripleContainer* outerTable = (tripleContainer*) malloc(outerSize);
		
 
		//Inizialize inner table
		for (int i = 0; i <INNER_SIZE - 1; i++ ) {
			innerTable[i] = {0, 180+i, 100*i, 0};
		}
		innerTable[INNER_SIZE -1] = {5, 200, 32, 0};


                //Inizialize outer table
                for (int i = 0; i <OUTER_SIZE - 1; i++ ) {
                        outerTable[i] = {0, 200+i, 200*i, 0};
                }
                outerTable[INNER_SIZE -1] = {9, 200, 32, 0};


		standard_context_t context;

		mem_t<tripleContainer> d_innerTable(INNER_SIZE, context);
		mem_t<tripleContainer> d_outerTable(OUTER_SIZE, context);
		cudaMemcpy(d_innerTable.data(), innerTable, innerSize, cudaMemcpyHostToDevice);
                cudaMemcpy(d_outerTable.data(), outerTable, outerSize, cudaMemcpyHostToDevice);

	
		cout << "starting join on the subject" << endl;
		TripleComparator comparator;
		mem_t<int2> joinResult = inner_join<empty_t, tripleContainer*, tripleContainer*, TripleComparator >(d_innerTable.data(), INNER_SIZE, d_outerTable.data(), OUTER_SIZE, comparator, context);
					
		std::vector<int2> result = from_mem(joinResult);

		for (int i =0; i <result.size(); i++) {
			std::cout << result[i].x << " " << result[i].y << std::endl;
		}	
		

		
	}
	
}


