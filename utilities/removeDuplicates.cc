#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>    // std::sort


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
		triple.push_back(strInput);
	}
	int oldSize = triple.size();
	std::sort(triple.begin(), triple.end()); 
  	auto last = std::unique(triple.begin(), triple.end());
   	triple.erase(last, triple.end());  	
	std::random_shuffle ( triple.begin(), triple.end() );
	int diff = (oldSize - triple.size());
	std::cout << "erased " << diff  << std::endl;
	
	for (string value : triple) {
		output << value << endl;
	}
}
