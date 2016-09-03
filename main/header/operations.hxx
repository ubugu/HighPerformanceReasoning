#pragma once

#include <cstdlib>

#include "types.hxx"

class Operation {

	protected:
		Binding* result_;	
		std::vector<std::string> variables_;

	public:
		Binding* getResult() {
			return result_;
		}
	
		void setResult(Binding* result) {
			result_ = result;
		}
			
		std::vector<std::string> getVariables() {
			return variables_;
		}
	
		Binding** getResultAddress() {
			return &result_;
		}
};
