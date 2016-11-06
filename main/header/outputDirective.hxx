#pragma once

#include <cstdlib>
#include <unordered_map>

#include "types.hxx"

//General class for output creation
class OutputDirective {	
	public:
		OutputDirective() {}
		
		//Function for generating the output
		virtual void generateOutput(std::unordered_map<size_t, Lit> mapH, RelationTable input) = 0;
};

//Class for SELECT directive for output formatting
class SelectDirective : public OutputDirective{
	private:
		//Projection variables. Assumed to be ordered in the same way of output table
		std::vector<int> variablesProjection_;
	
	public:
		//Constructor for SELECT with defined variables
		SelectDirective(std::vector<int> variablesProjection) : variablesProjection_(variablesProjection) {}
	
		//Constructur for SELECT with * element
		SelectDirective() {}
	
		//Function for generating the output
		void generateOutput(std::unordered_map<size_t, Lit> mapH, RelationTable input) {
			std::cout << "Found " << input.height << " elements" << std::endl;
			
			std::vector<std::string> variables = input.header;
			size_t* final_relationalTable;
			
			//Copy element from device memory or from main memory
			if (input.onDevice) {	
				final_relationalTable = (size_t*) malloc(input.height * input.width * sizeof(size_t));
				cudaMemcpy(final_relationalTable, input.pointer,  sizeof(size_t)*  input.height * input.width , cudaMemcpyDeviceToHost);
			} else {
				final_relationalTable = input.pointer;
			}	
			
			std::vector<std::string> output;
			//Dehash and save variable in the output vector;
			if ( variablesProjection_.size() == 0) {
				for (int i =0; i < input.height; i++) {
					for (int k = 0; k < input.width; k++) {
						output.push_back(mapH[final_relationalTable[i * input.width + k]].stringValue );
					//	std::cout << variables[k] << "=" <<   mapH[final_relationalTable[i * input.width + k]].stringValue << " ";
					}
					//std::cout << std::endl;
				}
			} else {
				for (int i =0; i < input.height; i++) {
					for (int k = 0; k < variablesProjection_.size(); k++) {
						int currentIndex = variablesProjection_[k];
						output.push_back(mapH[final_relationalTable[i * input.width + currentIndex ]].stringValue );
					//	std::cout << variables[currentIndex] << "=" <<   mapH[final_relationalTable[i * input.width + currentIndex]].stringValue  << " ";
					}
					//std::cout << std::endl;
				}
			}
	
			//Test value to be removed
			VALUE += input.height;
		}
};

//Class for ASK directive for output formatting
class AskDirective {


};


/*
struct OutGraph {
	//Number of triple * 3
	int pattern_size;
	//HASH MAP PER VARIBILI
	//HASH   
	//Vettore Preallocato contenente gia le variabili costanti. Devo solo metttere nei posti giusti le varaibili.
	std::vector<std::string> graphPattern;
}*/





//Class for CONSTRUCT directive for output formatting
class ConstructDirective {



};
