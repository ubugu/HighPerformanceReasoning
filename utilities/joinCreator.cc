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
	if (argc < 4) {
		cout << "Error number of elements. The format is: number of select, store size, different elements" << endl;
		exit(1);
	}
	
	std::string stringa(argv[3]);
	
        ofstream outf("../rdfStore/JoinTest/join" + stringa + ".txt");
        
        std::cout << "CRATED IN " << "../rdfStore/JoinTest/join" + stringa + ".txt" << std::endl;

	string strInput;
	srand (time(NULL));
	
	const int VALUE = 5001;

	int totalNumber = stoi(argv[1]);
	
	const int STORESIZE = stoi(argv[2]);
	int blockSize = STORESIZE / totalNumber;
	std::cout << blockSize << std::endl;
	std::vector<int> targetValue;
	std::vector<int> targetValue2;
	for (int i = 0; i < totalNumber; i++) {
		int val = rand() % blockSize + 1;
		if (i == 0) {
			targetValue.push_back(val);
		} else { 
			targetValue.push_back(val + targetValue[i  -1]);		
		}
		
		val = rand() % blockSize + 1;
		
		if (i == 0) {
			targetValue2.push_back(val);
		} else { 
			targetValue2.push_back(val + targetValue2[i  -1]);		
		}
		
	}


	
	int MAX_VALUE = atoi(argv[3]);

	int currentValue = 0;
	int currentValue2 = 0;
	vector<string> triple;
	
	for (int i =0; i < STORESIZE; i++) {
		std::stringstream ss;
		if (((currentValue != totalNumber) && (i == targetValue[currentValue])) && ((currentValue2 != totalNumber) && (i == targetValue2[currentValue2]))) { 
			ss << "<http://example.org/int/80001> <http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/90001> .";  
			triple.push_back(ss.str());
			currentValue2++;
			currentValue++;		
		} else  if ((currentValue != totalNumber) && (i == targetValue[currentValue])) {
			ss << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/90001> ."; 
			triple.push_back(ss.str());
			currentValue++;
		} else if ((currentValue2 != totalNumber) && (i == targetValue2[currentValue2])) {
			ss << "<http://example.org/int/80001> <http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE <<  "> ."; 
			triple.push_back(ss.str());
			currentValue2++;
		} else {
			ss << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> .";
			triple.push_back(ss.str());	
			
		}
	}
	
	std::cout << "current value1 " << currentValue << " " << currentValue2 << std::endl;
	
	std::sort(triple.begin(), triple.end()); 
  	auto last = std::unique(triple.begin(), triple.end());
   	triple.erase(last, triple.end());  	
	std::random_shuffle ( triple.begin(), triple.end() );


	std::cout << "STORE SIZE IS " << triple.size() << std::endl;
	
	for (int i=0; i <triple.size(); i++) {
		outf << triple[i] << " " << (i + 1) *1000 << std::endl;
	}

}
