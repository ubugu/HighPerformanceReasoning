#include <fstream>
#include <iostream>
#include <cstdlib>


int main(int argc,char *argv[]) 
{
	using namespace std;
	//THE FORMAT IS: NUMBER OF TRIPLE, MAX VALUE, PATH

	ofstream outf(argv[3]);

	const  int MAX_VALUE = atoi(argv[2]);
	const int TRIPLE_NUMBER = atoi(argv[1]);
	
	for (int i =0; i < TRIPLE_NUMBER; i ++)
		outf << rand() % MAX_VALUE << " " << rand() % MAX_VALUE << " " << rand() % MAX_VALUE<< " ." <<  endl;
	outf.close();
}
