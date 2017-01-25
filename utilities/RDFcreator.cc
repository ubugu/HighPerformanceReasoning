#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <algorithm> 
#include <vector>

int main(int argc,char *argv[]) 
{
	using namespace std;
	//THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, PATH

	if (argc < 4) {
		cout << "Wrong number of input elements. THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, PATH." << endl;
		return -1; 
	}
	
	ofstream outf(argv[3]);
	const  int MAX_VALUE = atoi(argv[2]);
	const int TRIPLE_NUMBER = atoi(argv[1]);
	/* initialize random seed: */
	srand (time(NULL));
	
	std::cout << "TRIPLE " << TRIPLE_NUMBER << " MAX VALUE " << MAX_VALUE << std::endl;

	vector<string> triple;

	for (int i =0; i < TRIPLE_NUMBER; i ++) {
		std::stringstream ss;
		ss << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> " << "<http://example.org/int/"   << rand() % MAX_VALUE << "> .";
		triple.push_back(ss.str());
	}
	

	std::sort(triple.begin(), triple.end()); 
  	auto last = std::unique(triple.begin(), triple.end());
   	triple.erase(last, triple.end());  	
	std::random_shuffle ( triple.begin(), triple.end() );


	std::cout << "STORE SIZE IS " << triple.size() << std::endl;
	
	for (int i=0; i <triple.size(); i++) {
		outf << triple[i] << " " << (i + 1) *1000 << std::endl;
	}
	
	outf.close();
}
