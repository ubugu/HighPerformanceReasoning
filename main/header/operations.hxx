#pragma once

#include <cstdlib>

#include "types.hxx"

class Operation {

	protected:
		RelationTable result_;	

	public:
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
		
		virtual void execute() = 0;
};
