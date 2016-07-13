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
	}
	
       fstream rdfStore (argv[1]);
       ofstream outf(argv[2]);

	//Format is: <http://example.org/int/553>	

	string strInput;
	srand (time(NULL));
	int secret = rand() % 100 + 1;
	while (getline(rdfStore,strInput)) {
		vector<string> triple;
        	separateWords(strInput, triple, ' ');
		
		outf << "<http://example.org/int/" + triple[0] + ">" << " " << "<http://example.org/int/" + triple[1] + ">" << " " << "<http://example.org/int/" + triple[2] + ">" << " ." << endl; 
	}
}
