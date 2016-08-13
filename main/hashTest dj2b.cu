#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include<vector>


//struct to contains a single triple with int type.
template<typename type_t>
struct tripleContainer {
        type_t subject;
        type_t predicate;
        type_t object;
};

/*
__device__ unsigned long hashF(char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}*/

/*
__global__ void hashD(tripleContainer<char*>* src, tripleContainer<unsigned long>* dest, int maxSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= maxSize ) {
		return;
	}
	
	dest[index] = {	hashF(src[index].subject), hashF(src[index].predicate), hashF(src[index].object)};

}
*/


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
 
		using namespace std;
		
		struct timeval begin, end;

/*
		std::vector<float> timeCuVector;                
		std::vector<float> timeExVector;
*/
		ifstream rdfStoreFile (argv[1]);
		string strInput;
		int fileLength = 0;	 
		while (std::getline(rdfStoreFile, strInput)) {
			++fileLength;
		}
	
		rdfStoreFile.clear();
		rdfStoreFile.seekg(0, ios::beg);

                size_t rdfSize = fileLength  * sizeof(tripleContainer<char*>);
                tripleContainer<char*> h_rdfStore[300000];

                //read store from rdfStore
                for (int i = 0; i <fileLength; i++) {
			getline(rdfStoreFile,strInput);
                        std::vector<string> triple;
                        separateWords(strInput, triple, ' ');
			
			
			h_rdfStore[i].subject = const_cast<char*> (triple[0].c_str());
                        h_rdfStore[i].predicate =  const_cast<char*>  (triple[1].c_str());
                        h_rdfStore[i].object = const_cast<char*>  (triple[2].c_str());

                }

                rdfStoreFile.close();
                
;
                tripleContainer<char*>* d_rdfStore;
                cudaMalloc(&d_rdfStore, rdfSize);
              	cudaMemcpy(d_rdfStore, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);
          	cudaDeviceSynchronize();			
         
                
                gettimeofday(&begin, NULL);		
              	tripleContainer< unsigned long>* d_hash;
              	cudaMalloc(&d_hash, fileLength * sizeof(tripleContainer<unsigned long>*));
              	
                hashD<<<30,1000>>>(d_rdfStore, d_hash, 300000);
       		gettimeofday(&end, NULL);
		
		
		float elapsed = 1000 * (end.tv_sec - begin.tv_sec) + (float) (end.tv_usec - begin.tv_usec) / (float) 1000;
		std::cout << "Elapsed time is " << elapsed  << std::endl;
		

		
		
                return 0;
}
