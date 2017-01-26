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
	
			//Dehash and save variable in the output vector; 
			//TODO output print is currently commented.
			if ( variablesProjection_.size() == 0) {
				for (int i =0; i < input.height; i++) {
					for (int k = 0; k < input.width; k++) {
						const char* value = mapH[final_relationalTable[i * input.width + k]].stringValue.c_str() ;

						//std::cout << variables[k] << "=" <<   mapH[final_relationalTable[i * input.width + k]].stringValue << " ";
					}
					//std::cout << std::endl;
				}
			} else {
				for (int i =0; i < input.height; i++) {
					for (int k = 0; k < variablesProjection_.size(); k++) {
						int currentIndex = variablesProjection_[k];
						(mapH[final_relationalTable[i * input.width + currentIndex ]].stringValue );
						//std::cout << variables[currentIndex] << "=" <<   mapH[final_relationalTable[i * input.width + currentIndex]].stringValue  << " ";
					}
					//std::cout << std::endl;
				}
			}


		}

};

