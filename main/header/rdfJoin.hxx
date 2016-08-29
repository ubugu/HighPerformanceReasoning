#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include "types.hxx"


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
		
		int innerMask[3];
		int outerMask[3];

	public:
		JoinOperation(Binding** innerTable, Binding** outerTable, int innerMask[3], int outerMask[3], std::vector<std::string> variables) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			this->variables = variables;
			std::copy(innerMask, innerMask + 3, this->innerMask);
			std::copy(outerMask, outerMask + 3, this->outerMask);
		};
};


void rdfJoin(Binding* innerTable, Binding* outerTable, std::vector<std::string> joinMask) {
	return;
}

