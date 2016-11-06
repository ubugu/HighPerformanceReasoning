#pragma once

#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "rdfOptional.hxx"
#include "rdfSelect.hxx"
#include "rdfJoin.hxx"
#include "rdfUnion.hxx"
#include "rdfFilter.hxx"
#include "outputDirective.hxx"

using google::dense_hash_map;

const SelectArr ELEMENT_VALUE[3] = {SelectArr::S, SelectArr::P, SelectArr::O};

//Get next character different from the white space
char* nextChar(char* pointer, char* end) {
	 while (pointer != end) {
		if (*pointer == ' ') {
			pointer++;
			continue;
		} else { 
			return pointer;
		}
	}
	
	return pointer;
}		

//Get the next word, untill the element character is reached. Intial white spaces are ignored.
std::string nextWord(char** pointer, char* end, const char element) {
	std::string word = "";
	
	*pointer = nextChar(*pointer, end);
	if (*pointer == end) {
		throw  std::string("Parsing error, unexpected token or end of string");
	}
	
	while(*pointer != end) {
		if (**pointer == element) {
			(*pointer)++;
			return word;
		}
		
		word += **pointer;
		(*pointer)++;
	}
	return word;
		
}

//Generate error message
std::string err(std::string expected, std::string found) {
	return	"Parsing error, expected " + expected + " found: '" + found + "'";
}

//Test if the string provided is equal to the one expected. Raise exception if they are not equal
bool testWord(std::string expected, std::string found) {
	if (found != expected) {
		throw err(expected, found);
	}
	return true;
}


//Parse time expresse by step and range in ms
unsigned long int timeParse(std:: string word) {
	char last = word[word.length() -1];

	unsigned long int time = 0;
	
	if (last == 's') {
		if (word[word.length() - 2] == 'm') {
			time = stoul(word.substr(0, word.length() -2));
		}
		
		else {
			time = 1000 * stoul(word.substr(0, word.length() -1));
		}	
	}
	
	else if (last == 'm') {
		time = 60 * (unsigned long int) 1000 * stoul(word.substr(0, word.length() -1));				
	}
	
	else if (last == 'h') {
		time = 60 * 60 * (unsigned long int) 1000 * stoul(word.substr(0, word.length() -1));					
	}
	
	else if (last == 'd') {
		time = 24 * 60 * 60 * (unsigned long int) 1000 * stoul(word.substr(0, word.length() -1));		
	}
	
	else {
		throw err("TIME UNIT", word);
	}
	
	return time;
			
}

/**
** Section for filter parser
**/


//Filter parser
BasicOp* filterParser(char** pointer, char* end, std::unordered_map<size_t, Lit>* hashMap, std::vector<std::string> variables) {
	std::string word;
	word = nextWord(pointer, end, ' ');
	//ISBOUND OP
	if ((word == "(") || (word == "!(")) {
		BasicOp* leftOp;
		if (word[0] == '!') {
			BasicOp* subOp = filterParser(pointer, end, hashMap, variables);
			NotOp* notop = new NotOp();
			notop->op = subOp;
			leftOp = notop;
		} else {
			leftOp = filterParser(pointer, end, hashMap, variables);
		}
		
		word = nextWord(pointer, end, ' ');
		testWord(")", word);
		
		char* temp;
		temp = *pointer;
		
		std::string booleanConnector = nextWord(&temp, end, ' ');
		
		if (booleanConnector == "&&") {
			*pointer = temp;
			word = nextWord(pointer, end, ' ');
			testWord("(", word);
			BasicOp* rightOp = filterParser(pointer, end, hashMap, variables);
			word = nextWord(pointer, end, ' ');
			testWord(")", word);
			AndOp* andop = new AndOp();
			andop->left = leftOp;
			andop->right = rightOp;
			return andop;
		} else if (booleanConnector == "||") {
			*pointer = temp;
			word = nextWord(pointer, end, ' ');
			testWord("(", word);
			BasicOp* rightOp = filterParser(pointer, end, hashMap, variables);
			word = nextWord(pointer, end, ' ');
			testWord(")", word);
			OrOp* orop = new OrOp();
			orop->left = leftOp;
			orop->right = rightOp;
			return orop;
		} else  {
			return leftOp;
		}
		 	
	} else if (word == "isBound"){
		//TODO is bound and ! is bound	
	} else {
		//Create variables for filter op
		int var[2] = {-1, -1};
		std::string input[2];
		std::string operation;
		std::vector<Lit> literalInput;
		
		input[0] = word;
		operation = nextWord(pointer, end, ' ');
		input[1] = nextWord(pointer, end, ' ');
		
		//Check if inputs are variables literals
		for (int i = 0; i <2; i++) {
			if (input[i][0] == '?') {
				for (int k =0; k < variables.size(); k++) {
					if (variables[k] == input[i]) {
						var[i] = k;
					}
				}
				Lit literal;
				literalInput.push_back(literal);
			} else {
				//Assuming string = "%string", numeric = %value% or "%vakue"^^%type%
				if (input[i][input[i].length()] != '"') {
					double value = std::stod(input[i]);
					Lit literal(value, Datatype::NUMERIC, "");
					literalInput.push_back(literal);
				} else {
					Lit literal = Lit::createLiteral(input[i]);
					literalInput.push_back(literal);
				}		
			}
		}
		
		if (operation == "<") {
			BinOp<Lit,Lit,Less>* op = new BinOp<Lit,Lit,Less>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;
		} else if (operation == "<=") {
			BinOp<Lit,Lit,LessEq>* op = new BinOp<Lit,Lit,LessEq>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;		
		} else if (operation == "=") {
			BinOp<Lit,Lit,Equal>* op = new BinOp<Lit,Lit,Equal>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;		
		} else if (operation == ">") {
			BinOp<Lit,Lit,Greater>* op = new BinOp<Lit,Lit,Greater>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;		
		} else if (operation == ">=") {
			BinOp<Lit,Lit,GreaterEq>* op = new BinOp<Lit,Lit,GreaterEq>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;
		} else if (operation == "!=") {
			BinOp<Lit,Lit,NotEq>* op = new BinOp<Lit,Lit,NotEq>();
			op->left = literalInput[0];
			op->right = literalInput[1];
			std::copy(var, var + 2, op->varIndex);
			return op;		
		} else  {
			std::cout <<  "ERROR EXPECTED BOOLEAN OPERATION " << std::endl;
			exit(-1);
		} 
	}
}
/**
** End filter parser section
**/		

//Parse a basic SPARQL block element, delimitated by { and }	
std::vector<Operation*>  blockElement(char** pointer, char* end,  CircularBuffer<size_t>* rdfPointer, std::unordered_map<size_t, Lit>* hashMap, std::unordered_map<std::string, size_t>* inverseHashMap) {		
	std::vector<Operation*> operationsVector;			
	std::string word;
	bool tripleEnded = true;
	std::vector<std::string> variable_stack;

	word = nextWord(pointer, end, ' ');
	while (word != "}") {
		//Code for subblock
		if (word == "{") {
			std::vector<Operation*>  subBlock = blockElement(pointer, end, rdfPointer, hashMap, inverseHashMap);
			
			if (subBlock.size() != 0) {
				std::vector<std::string> blockVariables = subBlock.back()->getVariables();
				
				if (variable_stack.size() == 0) {
					//Add operation as first select
					operationsVector.insert(operationsVector.end(), subBlock.begin(), subBlock.end());
					variable_stack = blockVariables;
				} else {
					//Variables for join
					std::vector<std::string> outerCopyvar;
					std::vector<int> innerIndex;
					std::vector<int> outerIndex;
					std::vector<int> outerCopyindex;
					bool found = false;
					
					//Calculate join variables
					for (int i = 0; i < blockVariables.size(); i++) {	
						for (int k = 0;  k < variable_stack.size(); k++) {
							if (variable_stack[k] == blockVariables[i]) {
								innerIndex.push_back(k);
								outerIndex.push_back(i);
								found = true;
								break;
							}
						}
						
						if (!found) {
							outerCopyindex.push_back(i);
							outerCopyvar.push_back(blockVariables[i]);
						}
					}
					
					if (outerCopyvar.size() == blockVariables.size()) {
						//TODO 
						std::cout << "NOT IMPLEMENTED YET, VARIABLE MUST APPEAR IN ORDER." << std::endl;
						exit(-1);	
					} else {
						variable_stack.insert(variable_stack.end(), outerCopyvar.begin(), outerCopyvar.end());
						JoinOperation* joinop = new JoinOperation(operationsVector.back()->getResultAddress(), subBlock.back()->getResultAddress(),  innerIndex, outerIndex, outerCopyindex, variable_stack);
						
						operationsVector.insert(operationsVector.end(), subBlock.begin(), subBlock.end());
						operationsVector.push_back(joinop);
					}
				}
			} 
			  
			word = nextWord(pointer, end, ' ');
			tripleEnded = true;
			continue;
		
		//Code for OPTIONAL evaluation
		} else if (word == "OPTIONAL") {
			word = nextWord(pointer, end, ' ');
			testWord("{", word);
			
			std::vector<Operation*>  subBlock = blockElement(pointer, end, rdfPointer, hashMap, inverseHashMap);
			
			if (subBlock.size() != 0) {
				std::vector<std::string> blockVariables = subBlock.back()->getVariables();
				
				if (variable_stack.size() == 0) {
					operationsVector.insert(operationsVector.end(), subBlock.begin(), subBlock.end());
					variable_stack = blockVariables;
				} else {
					//Variables for join
					std::vector<std::string> outerCopyvar;
					std::vector<int> innerIndex;
					std::vector<int> outerIndex;
					std::vector<int> outerCopyindex;
					bool found = false;
					
					//Calculate join variables
					for (int i = 0; i < blockVariables.size(); i++) {	
						for (int k = 0;  k < variable_stack.size(); k++) {
							if (variable_stack[k] == blockVariables[i]) {
								innerIndex.push_back(k);
								outerIndex.push_back(i);
								found = true;
								break;
							}
						}
						
						if (!found) {
							outerCopyindex.push_back(i);
							outerCopyvar.push_back(blockVariables[i]);
						}
						
						found = false;
					}
					
					if (outerCopyvar.size() == blockVariables.size()) {
						//TODO 
						std::cout << "NOT IMPLEMENTED YET, VARIABLE MUST APPEAR IN ORDER." << std::endl;
						exit(-1);	
					} else {
						variable_stack.insert(variable_stack.end(), outerCopyvar.begin(), outerCopyvar.end());
						OptionalOperation* optionalop = new OptionalOperation(operationsVector.back()->getResultAddress(), subBlock.back()->getResultAddress(),  innerIndex, outerIndex, outerCopyindex, variable_stack);
						operationsVector.insert(operationsVector.end(), subBlock.begin(), subBlock.end());
						operationsVector.push_back(optionalop);
					}
				}
			} 
			  
			word = nextWord(pointer, end, ' ');
			tripleEnded = true;
			continue;
		
		//Code for FILTER evaluation
		} else if (word == "FILTER") {
			//TODO FILTER CODE
			word = nextWord(pointer, end, ' ');
			testWord("(", word);
			BasicOp* filter = filterParser(pointer, end, hashMap, variable_stack);
			
			FilterOperation* filterOp = new FilterOperation(operationsVector.back()->getResultAddress(), filter, operationsVector.back()->getVariables(), hashMap);
			
			operationsVector.push_back(filterOp);			
			
			word = nextWord(pointer, end, ' ');
			testWord(")", word);
			word = nextWord(pointer, end, ' ');
			tripleEnded = true;
			continue;
		
		//Default case is triple pattern
		} else if (tripleEnded) {
			//Varaibles for select
			std::vector<std::size_t> constants;
			std::vector<std::string> selectVariable;
			int arr = 0;
		
			//Variables for join
			std::vector<std::string> outerCopyvar;
			std::vector<int> innerIndex;
			std::vector<int> outerIndex;
			std::vector<int> outerCopyindex;
		
			for (int i = 0; i <3; i ++ ) {
				//Check if it is a variable, blank node or constant
				if (word[0] == '?' || word[0] == '_') {
					selectVariable.push_back(word);

					bool found = false;
				
					for (int k = 0;  k < variable_stack.size(); k++) {
						if (variable_stack[k] == word) {
							innerIndex.push_back(k);
							outerIndex.push_back(selectVariable.size() - 1);
							found = true;
							break;
						}
					}
				
					if (!found) {
						outerCopyindex.push_back(selectVariable.size() - 1);
						outerCopyvar.push_back(word);
					}
											
				} else {
	
					size_t hashValue; 
					
					//Check if it has been already hashed
					if ((*inverseHashMap)[word] != 0) {
						hashValue = (*inverseHashMap)[word];
					} else {
						//Calculate hash value
						hashValue = hashFunction(word);
					
						//value 0 is reserved for unbound value
						hashValue = (hashValue == 0 ? hashValue + 1 : hashValue);
					
						//Check for collisions
						while ((*hashMap)[hashValue].stringValue != "") {
							hashValue++;
						}
						
						(*hashMap)[hashValue] = Lit::createLiteral(word);
						(*inverseHashMap)[word] = hashValue;					
					}
					
					constants.push_back(hashValue);
					arr += static_cast<int> (ELEMENT_VALUE[i]);   
				}

				word = nextWord(pointer, end, ' ');
			}
		
		
			SelectOperation* currentselect = new SelectOperation(constants, selectVariable, arr);
			currentselect->setStorePointer(rdfPointer);
		

			//Create join operation
			if (variable_stack.size() == 0) {
				operationsVector.push_back(currentselect);
				variable_stack.insert(variable_stack.end(), selectVariable.begin(), selectVariable.end());
			} else {
			
				if (outerCopyvar.size() == selectVariable.size()) {
					//TODO 
					std::cout << "NOT IMPLEMENTED YET, VARIABLE MUST APPEAR IN ORDER." << std::endl;
					exit(-1);	
				} else {
					variable_stack.insert(variable_stack.end(), outerCopyvar.begin(), outerCopyvar.end());
					JoinOperation* joinop = new JoinOperation( operationsVector.back()->getResultAddress(), currentselect->getResultAddress(),  innerIndex, outerIndex, outerCopyindex, variable_stack);
					operationsVector.push_back(currentselect);
					operationsVector.push_back(joinop);
				}
			}
			
			if (word == "." ) {
				tripleEnded = true;
				word = nextWord(pointer, end, ' ');
			} else {
				tripleEnded = false;
			}
			
		} else {
			std::cout << "PARSING ERROR " << tripleEnded << std::endl;
			exit(-1);
		}
			
		
	} 
	
	//Check if there is a union pattern
	char* temp = *pointer;

	//Check if pointer is at the end of the query string
	temp = nextChar(temp, end);
	std::cout << temp << " END " << end;
	if (temp == end) {
		std::cout << "ESCO " << std::endl;
		return operationsVector; 
	}
	word = nextWord(&temp, end, ' ');
	

	if (word == "UNION") {
		*pointer = temp;
		word = nextWord(pointer, end, ' ');
		testWord("{", word);
		
		std::vector<Operation*>  subBlock = blockElement(pointer, end, rdfPointer, hashMap, inverseHashMap);
		if (subBlock.size() != 0) {
			std::vector<std::string> blockVariables = subBlock.back()->getVariables();
			
			std::vector<std::string> rightCopyvar;
			std::vector<int> unionIndex;
			int lastElementPosition = variable_stack.size();
			bool found = false;
	
			for (int i = 0; i < blockVariables.size(); i++) {
				for (int k = 0;  k < variable_stack.size(); k++) {
					if (variable_stack[k] == blockVariables[i]) {
						unionIndex.push_back(k);
						found = true;
						break;
					}
				}
		
				if (!found) {
					unionIndex.push_back(lastElementPosition);
					lastElementPosition++;
					rightCopyvar.push_back(blockVariables[i]);
				}
				
				found = false;
			}
		
			variable_stack.insert(variable_stack.end(), rightCopyvar.begin(), rightCopyvar.end());
			UnionOperation* unionop= new UnionOperation(operationsVector.back()->getResultAddress(), subBlock.back()->getResultAddress(),  unionIndex, variable_stack);
			operationsVector.insert(operationsVector.end(), subBlock.begin(), subBlock.end());
			operationsVector.push_back(unionop);
		}
	}
				
	return operationsVector;
}			
		

