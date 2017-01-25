#include <fstream>
#include <iostream>
#include <cstdlib>


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

	for (int i =0; i < TRIPLE_NUMBER; i ++)
		outf << "<http://example.org/int/"   << i % MAX_VALUE << "> " << "<http://example.org/int/"   << (2 * i) % MAX_VALUE << "> " << "<http://example.org/int/"   << ( 5 * i) % MAX_VALUE << "> . " << (i + 1) * 1000 << std::endl;
	outf.close();


}
