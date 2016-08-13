#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include "types.hxx"

using namespace mgpu;

/*
* Join enum to define order and which element to join
* NJ indicates a non-join value, so it is ignored during join and sorting
* So that it improves performance avoiding uneecessary conditional expression
*/
enum class JoinMask {NJ = -1, SBJ = 0, PRE = 1, OBJ = 2};

//Section for defining operation classes
class JoinOperation 
{	
	private:
		mem_t<tripleContainer>** innerTable;
		mem_t<tripleContainer>** outerTable;
		mem_t<tripleContainer>* innerResult = 0;
		mem_t<tripleContainer>* outerResult = 0;
		
		JoinMask innerMask[3];
		JoinMask outerMask[3];

	public:
		JoinOperation(mem_t<tripleContainer>** innerTable, mem_t<tripleContainer>** outerTable, JoinMask innerMask[3], JoinMask outerMask[3]) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			std::copy(innerMask, innerMask + 3, this->innerMask);
			std::copy(outerMask, outerMask + 3, this->outerMask);
		};
			
		mem_t<tripleContainer>** getInnerTable() {
			return this->innerTable;
		};
		
		mem_t<tripleContainer>** getOuterTable() {
			return this->outerTable;
		};
		
		JoinMask* getInnerMask() {
			return this->innerMask;
		};
		
		JoinMask* getOuterMask() {
			return this->outerMask;
		};
		
		mem_t<tripleContainer>* getInnerResult() {
			return this->innerResult;
		};
		
		void setInnerResult(mem_t<tripleContainer>* result) {
			this->innerResult = result;
		};
		
		mem_t<tripleContainer>** getInnerResultAddress() {
			return &innerResult;
		}
		
		mem_t<tripleContainer>* getOuterResult() {
			return this->outerResult;
		};
		
		void setOuterResult(mem_t<tripleContainer>* result) {
			this->outerResult = result;
		};
		
		mem_t<tripleContainer>** getOuterResultAddress() {
			return &outerResult;
		}		
};

enum class SelectArr { S = 0, P = 1, O = 2, SP = 3, SO = 4, PO = 5, SPO = 6};

class SelectOperation 
{
	private:
		mem_t<tripleContainer>* query;
		mem_t<tripleContainer>* result = 0;
		int arr;

	public:
		SelectOperation(mem_t<tripleContainer>* query, SelectArr arr) {
			this->query = query;	
			this->arr = static_cast<int> (arr);
		};

		int getArr() {
			return this-> arr;
		}
			
		mem_t<tripleContainer>* getQuery() {
			return this->query;
		};
		                                                                            
		mem_t<tripleContainer>* getResult() {
			return this->result;
		};
		
		void setResult(mem_t<tripleContainer>* result) {
			this->result = result;
		};
		
		mem_t<tripleContainer>** getResultAddress() {
			return &result;
		}
};

