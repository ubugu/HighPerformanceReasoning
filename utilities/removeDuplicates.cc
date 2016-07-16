#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>    // std::sort

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
		cout << "Error number of elements. The format is: inputfile, outputFile" << endl;
	}
	
        fstream rdfStore (argv[1]);
 	ofstream output(argv[2]);

	//Format is: <http://example.org/int/553>	

	string strInput;
	int found = 0;
	vector<string> triple;
	while (getline(rdfStore,strInput)) {
		
        	//separateWords(strInput, triple, ' ');
		
		triple.push_back(strInput);
	}
	std::sort(triple.begin(), triple.end()); 
  	auto last = std::unique(triple.begin(), triple.end());
   	triple.erase(last, triple.end());  	
	std::random_shuffle ( triple.begin(), triple.end() );
	
	for (string value : triple) {
		output << value << endl;
	}
}