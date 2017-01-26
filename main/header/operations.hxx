#pragma once

#include <cstdlib>

#include "types.hxx"

//Base operation class
class Operation {

	protected:
		//Result table
		RelationTable result_;	

	public:
		//Constructor for setting the variables of the result table
		Operation (std::vector<std::string> variables) {
			result_.header = variables; 
		}
		
		RelationTable getResult() {
			return result_;
		}
			
		std::vector<std::string> getVariables() {
			return result_.header;
		}
	
		RelationTable* getResultAddress() {
			return &result_;
		}
		
		//Virtual function for executing the requested function	
		virtual void execute() = 0;
};
