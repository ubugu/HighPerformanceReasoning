#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <algorithm> 

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


int main(int argc,char *argv[]) 
{
	using namespace std;
	//THE FORMAT IS: INPUT FILE, OUTPUT FILE
	if (argc < 3) {
		cout << "Error number of elements. The format is: number of select, store size" << endl;
		exit(1);
	}
	
	std::string stringa(argv[2]);
	
        ofstream outf("../rdfStore/Filter/filter" + stringa + ".txt");
        

	string strInput;
	srand (time(NULL));
	
	const int VALUE = 5001;

	int totalNumber = stoi(argv[1]);
	
	const int STORESIZE = stoi(argv[2]);
	int blockSize = STORESIZE / totalNumber;
	std::cout << blockSize << std::endl;
	std::vector<int> targetValue;
	for (int i = 0; i < totalNumber; i++) {
		int val = rand() % blockSize + 1;
		if (i == 0) {
			targetValue.push_back(val);
		} else { 
			targetValue.push_back(val + targetValue[i  -1]);		
		}
		
	}


	
	int MAX_VALUE = 2000;
	int INT_VALUE = 3000;
	
	int currentValue = 0;
	vector<string> triple;
	
	for (int i =0; i < STORESIZE; i++) {
		std::stringstream ss;
		if ((currentValue != totalNumber) && (i == targetValue[currentValue])) {
			ss << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> \"" << rand() % INT_VALUE << "\"^^<http://www.w3.org/2001/XMLSchema#integer> ."; 
			triple.push_back(ss.str());
			currentValue++;
		} else {
			ss << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> .";
			triple.push_back(ss.str());	
			
		}
	}
	
	
	
	std::sort(triple.begin(), triple.end()); 
  	auto last = std::unique(triple.begin(), triple.end());
   	triple.erase(last, triple.end());  	
	std::random_shuffle ( triple.begin(), triple.end() );


	std::cout << "STORE SIZE IS " << triple.size() << std::endl;
	
	for (int i=0; i <triple.size(); i++) {
		outf << triple[i] << " " << (i + 1) *1000 << std::endl;
	}

}
