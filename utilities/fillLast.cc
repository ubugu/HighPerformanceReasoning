#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include <vector>


int main(int argc, char** argv) {

		using namespace std;



		if (argc < 4) {
			std::cout << "Error format is : input, output, target number" << std::endl;
		}

		ifstream rdfStoreFile (argv[1]);
		ofstream out(argv[2]);

		string strInput;
		int target = atoi(argv[3]);
		int fileLength = 0;
		while (std::getline(rdfStoreFile, strInput)) {
			++fileLength;
		}
		rdfStoreFile.clear();
		rdfStoreFile.seekg(0, ios::beg);


                std::vector<std::string> triples;

                for (int i = 0; i <fileLength; i++) {
			getline(rdfStoreFile,strInput);
                        triples.push_back(strInput);
                }

		std::cout << "SIZE BEFORE IS " << triples.size() << std::endl;

		int toFill = target - triples.size();
		std::cout << "NEED TO FILL " << toFill << std::endl;

		srand (time(NULL));
		for (int i =0; i < toFill; i ++)
			triples.push_back(to_string(rand() % 500) + " " + to_string(rand() % 500) + " " +  to_string(rand() % 500) +  " .");


                std::cout << "SIZE AFTER IS " << triples.size() << std::endl;


		for (int i=0; i < triples.size(); i++) {
			out << triples[i] << std::endl;
		}
		out.close();

                rdfStoreFile.close();


}
