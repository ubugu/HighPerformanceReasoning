#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>



//struct to contains a single triple.
struct tripleContainer {
        int subject;
        int predicate;
        int object;
};

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
	//THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, PATH


	


        //read store from file

       fstream rdfStore ("../rdfStore/rdf2.txt");

       string strInput;
       const int FILE_LENGHT = 10000;

       for (int i = 0; i < FILE_LENGHT; i++) {
		getline(rdfStore,strInput);
       		cout << i << endl;
        	std::vector<string> triple;
        	separateWords(strInput, triple, ' ');

		cout << separateWords[0] << endl;
		break
//        	tripleContainer newTriple {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi( triple[2].c_str())};
	}

       

	ofstream outf("../rdfStore/rdf2Int.txt");

	;
}
