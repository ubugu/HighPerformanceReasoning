#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
 
#include "header/manager.hxx"
#include "header/types.hxx"



//Function for parsing the input file
int separateWords(std::string inputString, std::vector<std::string> &wordVector, const char separator, int limit) {	
	const size_t zeroIndex = 0;
	size_t splitIndex = inputString.find(separator);
	while (splitIndex != -1) {
		wordVector.push_back(inputString.substr(zeroIndex, splitIndex));	
		inputString = inputString.substr(splitIndex + 1 , inputString.length() - 1);
		splitIndex = inputString.find(separator);
	}
	
	wordVector.push_back(inputString);
	return 0;
}




int main(int argc, char** argv) {

	using namespace std;	

	if (argc < 3) {
		std::cout << "Error program need two argumens: input_file, query" << std::endl;
		return -1;
	}

	//READ STORE FROM FILE
	ifstream rdfStoreFile (argv[1]);
	string strInput;

	//Calculate store lenght
	int fileLength = 0;	 
	while (std::	getline(rdfStoreFile, strInput)) {
		++fileLength;
	}
	std::cout << "STORE SIZE IS " << fileLength << std::endl;
	rdfStoreFile.clear();
	rdfStoreFile.seekg(0, ios::beg);
	
	//Cope elements from disk to main memory
	size_t rdfSize = fileLength  * sizeof(std::string) * 4;
	std::string* h_rdfStore = (std::string*) malloc(rdfSize);
	for (int i = 0; i <fileLength; i++) {
		getline(rdfStoreFile,strInput);
		std::vector<std::string> triple;
		separateWords(strInput, triple, ' ',5);
		
		h_rdfStore[i * 4 + 0]  = triple[0];
		h_rdfStore[i * 4 + 1]  = triple[1];
			
		h_rdfStore[i * 4 + 2]  = triple[2];
		h_rdfStore[i * 4 + 3]  = triple[4];	
	}
	rdfStoreFile.close();
	//END RDF READ


	//Action for GPU startup
	int* test;
	cudaMalloc(&test, sizeof(int));

	//Declare buffer size
	size_t buffer_size = 2000000;

	//Create query manager specifying the input stream and its length, and the size of the circular buffer to use	
	QueryManager manager(h_rdfStore, fileLength, buffer_size);

	//Parse query. Manage exception raised by the query parsers
	try {	
		manager.parseQuery(argv[2]);		
	} catch (std::string exc) {
		std::cout << "Exception raised: " << exc << std::endl;
		exit(1);
	}
	
	//Start query execution.
	manager.start();
}

