#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include <sparsehash/dense_hash_map>
#include <sys/time.h>

#include "header/manager.hxx"


using google::dense_hash_map;



//TODO implementare la projection su gpu
//TODO PASSARE I TIMESTAMP DA US A MS (OVERFLOW DI DAY E HOUR)


template<typename type_t, typename accuracy>
std::vector<accuracy> stats(std::vector<type_t> input) {
	int size = input.size();
	float mean = 0;
	float variance = 0;
	for (int i = 0; i < size; i++) {
		mean += (accuracy) input[i];
                variance += (accuracy)  (input[i] * input[i]);
        }
        mean = mean / ((accuracy) size);
        variance = variance / ((accuracy) size);
        variance = variance - (mean * mean);
        std::vector<accuracy> statistic;
	statistic.push_back(mean);
	statistic.push_back(variance);
	return statistic;
}



int main(int argc, char** argv) {
 
	using namespace std;

	std::vector<float> timeCuVector;                	
	std::vector<long int> timeExVector;	

	//READ STORE FROM FILE
	ifstream rdfStoreFile (argv[1]);
	string strInput;

	int fileLength = 0;	 
	while (std::getline(rdfStoreFile, strInput)) {
		++fileLength;
	}
	
	std::cout << "STORE SIZE IS " << fileLength << std::endl;
		
	rdfStoreFile.clear();
	rdfStoreFile.seekg(0, ios::beg);

	size_t rdfSize = fileLength  * sizeof(std::string);
	std::string* h_rdfStore = (std::string*) malloc(rdfSize);

	for (int i = 0; i <fileLength; i++) {
		getline(rdfStoreFile,strInput);
	    h_rdfStore[i]  = strInput;
	}
    rdfStoreFile.close();
	//END RDF READ

	struct timeval beginT, endT;
	
	cudaDeviceReset();

    size_t BUFFER_SIZE = 400000;
  
    int N_CYCLE = 1;

	for (int i = 0; i < N_CYCLE; i++) {

		gettimeofday(&beginT, NULL);

		QueryManager manager(h_rdfStore, fileLength, BUFFER_SIZE);
					
		try {
			//TODO controllare errore se manca sapazio finale? 
		//	manager.parseQuery("FROM STREAM <streamUri> RANGE TRIPLES 7000 SELECT ?s WHERE { ?p ?s <http://example.org/int/9> . <http://example.org/int/90> ?w ?p } ");
			manager.parseQuery(argv[2]);
		}
		catch (std::string exc) {
			std::cout << "Exception raised: " << exc << std::endl;
			exit(1);
		}
		
		manager.start();
		cudaDeviceSynchronize();
		gettimeofday(&endT, NULL);

		float exTime = (endT.tv_sec - beginT.tv_sec ) * 1000 + ((float) endT.tv_usec - (float) beginT.tv_usec) / 1000 ;
					
		timeCuVector.push_back(exTime);

		cout << "Time: " << exTime << endl;
	}

	std::vector<float> statistics = stats<float, float>(timeCuVector);	
    cout << "mean cuda time " << statistics[0] << endl;
    cout << "variance cuda time " << statistics[1] << endl;
	cout << "FINAL VALUE IS " << VALUE << std::endl;
				
    return 0;
}

