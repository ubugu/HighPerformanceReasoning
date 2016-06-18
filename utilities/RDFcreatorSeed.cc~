#include <fstream>
#include <iostream>
#include <cstdlib>


int main(int argc,char *argv[]) 
{
	using namespace std;
	//THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, SEEDNUMBER, SEEDVALUE, PATH,

	if (argc < 6) {
		cout << "Wrong number of input elements. THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, PATH." << endl;
		return -1; 
	}
	
	ofstream outf(argv[5]);
	const  int MAX_VALUE = atoi(argv[2]);
	const int TRIPLE_NUMBER = atoi(argv[1]);
	/* initialize random seed: */
	
	srand (time(NULL));
	int seedNumber = atoi(argv[3]);
	int seedValue = atoi(argv[4]);
	for (int i =0; i < TRIPLE_NUMBER; i ++) {
		if (i %  seedNumber == 0) {
			outf << seedValue << " " << rand() % MAX_VALUE << " " << rand() % MAX_VALUE<< " ." <<  endl;		
		} else {
			outf << rand() % MAX_VALUE << " " << rand() % MAX_VALUE << " " << rand() % MAX_VALUE<< " ." <<  endl;
		}
	}
	
	outf.close();
}
