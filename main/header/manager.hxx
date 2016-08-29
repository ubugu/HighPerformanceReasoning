#pragma once

#include <cstdlib>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "query.hxx"


using google::dense_hash_map;

int separateWords(std::string inputString, std::vector<std::string> &wordVector,const char separator ) {	
	const size_t zeroIndex = 0;
	size_t splitIndex = inputString.find(separator);
	
	while (splitIndex != -1)
		{
			wordVector.push_back(inputString.substr(zeroIndex, splitIndex));	
			inputString = inputString.substr(splitIndex + 1 , inputString.length() - 1);
			splitIndex = inputString.find(separator);
		}
	
	wordVector.push_back(inputString);
	return 0;
}





class QueryManager {
	private:
		std::string* source;
		int srcSize;
		
		std::vector<TimeQuery> time_queries_;
		std::vector<CountQuery> count_queries_;
		std::vector<TripleContainer> storebuffer_;
		
		CircularBuffer<long int> timestamp_pointer_;
		CircularBuffer<TripleContainer> storepointer_;	
    		dense_hash_map<size_t, std::string> resourcemap_;
		
	public:
		QueryManager(std::string* source, int srcSize, int buffSize)   {
			this->srcSize = srcSize;
			this->source = source;
			
			timestamp_pointer_.pointer = (long int*) malloc(buffSize * sizeof(long int));
			timestamp_pointer_.size = buffSize;
			
			cudaMalloc(&storepointer_.pointer, buffSize * sizeof(TripleContainer));
			storepointer_.size = buffSize;
			
			resourcemap_.set_empty_key(NULL);      
		}


		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan + storepointer_.pointer, &storebuffer_[0] + hostSpan, copySize * sizeof(TripleContainer), cudaMemcpyHostToDevice); 
		}

		void advancestorepointer_() {
			int copySize = storebuffer_.size();
			
			CircularBuffer<TripleContainer> rdfBuff = storepointer_;

			int newEnd = (rdfBuff.end + copySize) % rdfBuff.size;
	 
			if (newEnd < rdfBuff.end) {
				int finalHalf = rdfBuff.size - rdfBuff.end;
				copyElements(storepointer_.end, 0, finalHalf);			
	
				int firstHalf = copySize - finalHalf;
				copyElements(0, finalHalf, firstHalf);			
			} else {
				copyElements(storepointer_.end, 0, copySize);	
			}

			storepointer_.end = newEnd;

			storebuffer_.clear();
		}
		
		void checkStep() {	
			for (auto &query : count_queries_)  {
				query.incrementCount();
				if (query.isReady()) {
					advancestorepointer_();
					query.setWindowEnd(storepointer_.end);			
					query.launch();
					query.printResults(resourcemap_);
				}
			}
			
			for (auto &query : time_queries_) {
				if (query.isReady(timestamp_pointer_.pointer[timestamp_pointer_.end - 1])) {
					advancestorepointer_();
					query.setWindowEnd(storepointer_.end - 1);		
					query.launch();
					query.printResults(resourcemap_);
					query.setWindowEnd(1);
				}				
			}
		}
		
		void start() {
			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			long int ts = startingTs.tv_sec * 1000000 + startingTs.tv_usec;

			for (auto &query : time_queries_) {
				query.setStartingTimestamp(ts);
			}
			
			usleep(1);

			 
		

			for (int i =0; i <srcSize; i++) {

				
				TripleContainer currentTriple;
 
                                std::vector<std::string> triple;
                                separateWords(source[i], triple, ' ');
			
			        currentTriple.subject = h_func(triple[0]);
                                currentTriple.predicate = h_func(triple[1]);
                                currentTriple.object = h_func(triple[2]);

				resourcemap_[currentTriple.subject] = triple[0];
                                resourcemap_[currentTriple.predicate] = triple[1];
                                resourcemap_[currentTriple.object] = triple[2] ;

				struct timeval tp;
				gettimeofday(&tp, NULL);
				long int ms = tp.tv_sec * 1000000 + tp.tv_usec;


				timestamp_pointer_.pointer[timestamp_pointer_.end] = ms;
				timestamp_pointer_.end = (timestamp_pointer_.end + 1) % timestamp_pointer_.size;
				timestamp_pointer_.begin = timestamp_pointer_.end;
				
				storebuffer_.push_back(currentTriple);
							
				checkStep();

			}
			
			//TODO vedere se occorre tenere o no quest'ultima parte
			//***** REMOVE IN FINAL CODE: ONLY FOR TEST (FORSE) *****
			//DA OTTIMIZZARE POICHE VIENE LANCIATO ANCHE QUANDO NON SERVE!!
			advancestorepointer_();	 
			for (auto &query : time_queries_)  {
				query.setWindowEnd(storepointer_.end);
				query.launch();
				query.printResults(resourcemap_);
			}
			
			for (auto &query :count_queries_) {
				query.setWindowEnd(storepointer_.end);
				query.launch();
				query.printResults(resourcemap_);
			}
			//***** END REMOVE PART-*****			
		}
		
		
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
		
		
		
		std::string err(std::string expected, std::string found) {
			return	"Parsing error, expected " + expected + " found: '" + found + "'";
		}
		
		bool testWord(std::string expected, std::string found) {
			if (found != expected) {
				throw err(expected, found);
			}
			return true;
		}
		
		long int timeParse(std:: string word) {
			char last = word[word.length() -1];
			//U_SEC TIME
			long int time = 0;
			
			if (last == 's') {
				if (word[word.length() - 2] == 'm') {
					time = 1000 * stoi(word.substr(0, word.length() -2));
				}
				
				else {
					time = 1000000 * stoi(word.substr(0, word.length() -1));
				}	
			}
			
			else if (last == 'm') {
				time = 60 * (long int) 1000000 * stoi(word.substr(0, word.length() -1));				
			}
			
			else if (last == 'h') {
				time = 60 * 60 * (long int) 1000000 * stoi(word.substr(0, word.length() -1));					
			}
			
			else if (last == 'd') {
				time = 24 * 60 * 60 * (long int) 1000000 * stoi(word.substr(0, word.length() -1));		
			}
			
			else {
				throw err("TIME UNIT", word);
			}
			
			return time;
					
		}
		
		void parseQuery(std::string query) {
		
			//CHECK CHARACTER WORD 
			
			char* pointer = &query[0];
			char* end = &query[query.length() - 1];
			
			std::vector<std::string> variables;
			dense_hash_map<std::string, std::string> prefixes;
			prefixes.set_empty_key("");    			
			
			std::vector<SelectOperation*> selectOperations;
			std::vector<JoinOperation*> joinOperations;			
			dense_hash_map<std::string, Operation**> stackOperations;
			stackOperations.set_empty_key("");
			
			bool logical = false;
			long int window = 0;
			long int step = 0;
			
			
			bool addAll = false;
									
									
			std::string word = nextWord(&pointer, end, ' ');
			
			if (word == "BASE") {
			 	//IMPLEMENTATION OF BASE
			}
			
			while(word == "PREFIX") {
				std::string prefix = nextWord(&pointer, end, ':');
				std::string prefixUri = nextWord(&pointer, end, ' ');
				prefixes[prefix] = prefixUri;
				//CHECK DUPLICATE PREFIX
				word = nextWord(&pointer, end, ' ');				
			}
			
			
			
			testWord("FROM", word);
			word = nextWord(&pointer, end, ' ');				
			
			//'NAMED' TOKEN
		
			testWord("STREAM", word);
			
						
			//INDETIFY WHAT TO DO WITH STREAM IRI	
			std::string streamIri = nextWord(&pointer, end, ' ');				

			word = nextWord(&pointer, end, ' ');			
			testWord("RANGE", word);
			
			word = nextWord(&pointer, end, ' ');				
			
			if (word == "TRIPLES") {
				word = nextWord(&pointer, end, ' ');
				step = std::stoi(word);
				logical = false;
				//NEED TO CHECK IF NO INTEGER IS INSERTED
			}
			
			else {
				//VERIFICARE SE RICEVO UN INT SEGUITO DA UNIT° DI MISURA / SPAZIO VALIDO O NO
				window = timeParse(word);
				
				word = nextWord(&pointer, end, ' ');							
				
				if (word == "STEP") {
					word = nextWord(&pointer, end, ' ');							
					step = timeParse(word);
				}
				
				else if (word == "TUMBLING") {
					step = window;
				}
				
				else {
					throw err("STEP TIME", word);
				}
				
				
			}
			
			
			
			word = nextWord(&pointer, end, ' ');	
			
			
			if (word == "SELECT")  {

				word = nextWord(&pointer, end, ' ');
								
				do {					
					if (word == "*") {
						addAll = true;
						word = nextWord(&pointer, end, ' ');
						if (word != "WHERE") {
							throw err("WHERE", word); 
						} 
						break;
					}
					
					if (word[0] != '?') {
						throw std::string("Error in variable entered");
					}
					
					variables.push_back(word.substr(1));
					word = nextWord(&pointer, end, ' ');	
					
				} while (word != "WHERE");
				
				word = nextWord(&pointer, end, ' ');
				
				if (word != "{") {
					throw err("{", word);
				}
				
				
				do  {
					std::vector<std::size_t> constants;
					std::vector<std::string> selectVariable;
					const SelectArr stdArr[3] = {SelectArr::S, SelectArr::P, SelectArr::O};
					int arr = 0;
					
					for (int i = 0; i <3; i ++ ) {
						word = nextWord(&pointer, end, ' ');
						
						if (word[0] == '?') {
							if (addAll) {
								//CHECK FOR DUPLICATE VARIABLES
								variables.push_back(word.substr(1));
							}
							selectVariable.push_back(word.substr(1));
						} else {
							constants.push_back(h_func(word));
							arr += static_cast<int> (stdArr[i]);   
						}
						
					}
										
					SelectOperation* currentOp = new SelectOperation(constants, selectVariable, arr);
					selectOperations.push_back(currentOp);
					
					std::cout << "ADDED SELECT, total size is " << selectOperations.size() << std::endl;
					std::cout << "values are:" << std::endl;
					
					std::cout << "constants" << std::endl;
					for (int i = 0; i < constants.size(); i++) {
						std::cout << "- " << constants[i] << std::endl;
					}
					
					std::cout << "varaibles " << std::endl;
					
					for (int i =0; i<variables.size(); i++) {
						std::cout << "- " << variables[i] << std::endl;
					}
					
					std::cout << "ARR VALUS IS " << arr << std::endl;
					
					/**** SECTION FOR JOIN ***/
					dense_hash_map<Operation**, std::vector<std::pair<std::string, int>*>> currentStack;
					currentStack.set_empty_key(NULL);
				
					Operation* op = currentOp;
		
					int index = 0;
					for (auto var : selectVariable) {
						Operation** pair = stackOperations[var];
						
						if (!pair) {
							stackOperations[var] = &op;
						} else {
							currentStack[pair].push_back(new std::pair<std::string,int>(var, index));
						}
						
						index++;						
					}

					auto iter = currentStack.begin();	
					auto endIt = currentStack.end();
					while (iter != endIt) {
						Operation** pair = iter->first;
						auto pairVar = iter->second;
						
						int outerPos[3] = {-1, -1, -1};
						int innerPos[3] = {-1, -1, -1};
						
						for (int i = 0; i < pairVar.size(); i++) {
							innerPos[i] = pairVar[i]->second;
							
							int k = 0;
							for(std::string var : (*pair)->getVariables()) {
								if (pairVar[i]->first == var) {
									outerPos[i] = k;
									break;
								}
								k++;
							}	
						}
						         
						std::vector<std::string> joinedVariables = op->getVariables();
						
						JoinOperation* join = new JoinOperation(op->getResultAddress(), (*pair)->getResultAddress(), innerPos, outerPos, joinedVariables);
				 		iter++;
					}
					
					
					/**** END JOIN SECTION ***/

					word = nextWord(&pointer, end, ' ');
						
					if (word != "." && word != "}") {
						throw err(". or }", word);
					}
					
				} while (word != "}");
					
					
			} else {
			
				throw err("SELECT", word);
			}
				
								
			pointer = nextChar(pointer, end);
			
			if (pointer != end)  {
				throw std::string("Unexpected token");
			}
			
			
			if (logical == true) {
				TimeQuery query(selectOperations, joinOperations, storepointer_, timestamp_pointer_, window, step);
				time_queries_.push_back(query);
			}
			
			else {
				CountQuery query(selectOperations, joinOperations, storepointer_, window);
				count_queries_.push_back(query);						
			}				
		}
};


