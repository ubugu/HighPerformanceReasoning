#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>


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
       const int FILE_LENGHT = 299829;
        ofstream outf("../rdfStore/rdf2Int.txt");


       for (int i = 0; i < FILE_LENGHT; i++) {
		getline(rdfStore,strInput);
       		std::vector<string> triple;
        	separateWords(strInput, triple, ' ');

		std::vector<string> subject;
		separateWords(triple[0], subject, '/');
		subject[4].pop_back();
		
                std::vector<string> predicate;
                separateWords(triple[1], predicate, '/');
                predicate[4].pop_back();
                
                std::vector<string> object;
                separateWords(triple[2], object, '/');
                object[4].pop_back();
               
		outf << subject[4] << " " << predicate[4] << " " << object[4] << " ." << endl;
	}
}
