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
	int s1 = rand() % 100 + 1;
        int s2 = rand() % 100 + 1;
        int s3 = rand() % 100 + 1;

	string  secret1 = to_string(s1);
        string  secret2 = to_string(s2);
        string  secret3 = to_string(s3);


	while (getline(rdfStore,strInput)) {
		vector<string> triple;
        	separateWords(strInput, triple, ' ');

	        int s1 = rand() % 100 + 1;
        	int s2 = rand() % 100 + 1;
	        int s3 = rand() % 100 + 1;

        	string  secret1 = to_string(s1);
        	string  secret2 = to_string(s2);
       		string  secret3 = to_string(s3);
	
		

		outf << "<http://example.org/" +secret1 + "/" + triple[0] + ">" << " " << "<http://example.org/" + secret2 + "/" + triple[1] + ">" << " " << "<http://example.org/" + secret3 + "/" + triple[2] + ">" << " ." << endl; 
	}
}
