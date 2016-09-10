#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>


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
		exit(1);
	}
	
       fstream rdfStore (argv[1]);
       ofstream outf(argv[2]);

	string strInput;
	long int ts = 1;
	while (getline(rdfStore,strInput)) {		
		outf << strInput << " " << ts * 1000000 << std::endl; 
		ts++;
	}
}
