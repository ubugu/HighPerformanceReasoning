#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>

#include "types.hxx"
#include "operations.hxx"

/*
* Join enum to define order and which element to join
* NJ indicates a non-join value, so it is ignored during join and sorting
* So that it improves performance avoiding uneecessary conditional expression
*/
enum class JoinMask {NJ = -1, SBJ = 0, PRE = 1, OBJ = 2};


//Section for defining operation classes
class JoinOperation : public Operation
{	
	//TODO Modificare classe in modo che permetta la join di join
	private:
		Binding** innerTable;
		Binding** outerTable;
		
		int indexes[3];

	public:
		JoinOperation(Binding** innerTable, Binding** outerTable, int indexes[3], std::vector<std::string> variables) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			this->variables_ = variables;
			std::copy(indexes, indexes + 3, indexes);
		}
		
		
		void rdfJoin() {
			
			std::cout << "PRINTING TEST FOR JOIN" << std::endl;
			std::cout << "INNER ADDRESS " << *innerTable << " OUTER ADDRESS " << *outerTable <<  std::endl;
			std::cout << "INDEXES ARE " << indexes[0] << " " << indexes[1] << " " << indexes[2] << " " << std::endl;
			for (auto var : variables_) {
				std::cout << var << " - " ;
			}
			std::cout << std::endl;
			std::cout << "RESULT ADDRESS IS " << &result_ << std::endl;
			return;
		}
};





