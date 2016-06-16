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
	for (int i =0; i < TRIPLE_NUMBER; i ++)
		outf << rand() % MAX_VALUE << " " << rand() % MAX_VALUE << " " << rand() % MAX_VALUE<< " ." <<  endl;
	outf.close();
}
