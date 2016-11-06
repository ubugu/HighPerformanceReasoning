#pragma once

#include <cstdlib>
#include <vector>

#include "types.hxx"
#include "operations.hxx"


/**
** Struct for boolean comparison
**/
struct Equal {
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a == b);
	}
};


struct  Less {
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a < b);
	}
};

struct LessEq{
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a <= b);
	}
};

struct Greater {
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a > b);
	}
};


struct GreaterEq{
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a >= b);
	}
};

struct NotEq{
	template <typename type_t>
	bool operator ()(type_t a, type_t b) {
		return (a != b);
	}
};
/**
** END boolean struct
**/



/**
** Struct for filter operations
**/
struct BasicOp  {
	virtual bool  execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) = 0;
};


struct AndOp : BasicOp {
	BasicOp* left;
	BasicOp* right;
	
	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {
		return left->execute(hashMap, row) && right->execute(hashMap, row);
	
	}
};

struct OrOp : BasicOp {
	BasicOp* left;
	BasicOp* right;
	
	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {
		return left->execute(hashMap, row) || right->execute(hashMap, row);
	
	}
};

struct NotOp : BasicOp {
	BasicOp* op;
	
	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {
		return !op->execute(hashMap, row);
	
	}
};

struct isBound : BasicOp {
	int index;
	
	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {
		size_t hashValue = row[index];
		return hashValue != 0;
	}
};

template<typename t_1, typename t_2, typename operation>
struct BinOp : public BasicOp {
	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {		
		return false;

	}
};

template<typename operation_t>
struct BinOp<Lit, Lit, operation_t> : public BasicOp {
	Lit left;
	Lit right;

	int varIndex[2];

	bool execute(std::unordered_map<size_t, Lit>* hashMap, size_t* row) {		
		//Dehash left value if it is an hash one
		if (varIndex[0] != -1) {
			size_t hashValue = row[varIndex[0]];
			if (hashValue == 0) {
				return false;
			}
			left = (*hashMap)[hashValue];
		}
		
		//Dehash right value if it is an hash one
		if (varIndex[1] != -1) {
			size_t hashValue = row[varIndex[1]];
			if (hashValue == 0) {
				return false;
			}
			right = (*hashMap)[hashValue];
		}

		if(right.type != left.type) {
			return false;
		}
		
		operation_t op;
		switch (static_cast<int>( left.type)) {
			case(0):{
				std::string value1;
				std::string value2;
		
				value1 = left.stringValue;
				value2 = right.stringValue;
	
				return op(value1, value2);
			}
					
			case(1): {
				double value1 = left.numericValue;
				double value2 = right.numericValue;
			
				return op(value1, value2);	
			}
		}
	
		return false;
	}
};
/**
** End of struct for boolean comparison
**/


class FilterOperation : public Operation
{
	private:
		RelationTable* input_;
		BasicOp* filter_;
		std::unordered_map<size_t, Lit>* hashMap_;
		
	public:
		FilterOperation(RelationTable* inputTable, BasicOp* filter, std::vector<std::string> variables, std::unordered_map<size_t, Lit>* hashMap) : Operation(variables) {
			this->input_ = inputTable;
			this->filter_ = filter;
			this->hashMap_ = hashMap;
		}

		void execute() {
			//Copy input from device to host
			size_t* hostInput = (size_t*) malloc(sizeof(size_t) * input_->height * input_->width);
			cudaMemcpy(hostInput, input_->pointer, sizeof(size_t) * input_->height * input_->width, cudaMemcpyDeviceToHost);
			
			//Allocate result on host
			size_t* hostResult = (size_t*) malloc(sizeof(size_t) * input_->height * input_->width);
			int resultSize = 0;
			
			for (int i = 0; i <= input_->height; i++) {
				size_t* currentRow = hostInput + i * input_->width;
				
				if (filter_->execute(hashMap_, currentRow)) {
					std::copy(currentRow, currentRow + input_->width, hostResult + resultSize * input_->width);
					resultSize++;
				}
			}
			
			//allocate result on device and copy it from host to device
			result_.allocateOnDevice(resultSize);
			cudaMemcpy(result_.pointer, hostResult, sizeof(size_t) * result_.width * resultSize, cudaMemcpyHostToDevice);
			
			//Free unused memory
			cudaFree(input_->pointer);
			free(hostResult);
			free(hostInput);	
		}	
};


