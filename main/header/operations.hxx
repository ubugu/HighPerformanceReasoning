#pragma once

#include <cstdlib>

#include "types.hxx"

class Operation {

	protected:
		Binding* result;	
		std::vector<std::string> variables;

	public:
		Binding* getResult() {
			return this->result;
		}
	
		void setResult(Binding* result) {
			this->result = result;
		}
			
		std::vector<std::string> getVariables() {
			return variables;
		}
	
		Binding** getResultAddress() {
			return &result;
		}
};
