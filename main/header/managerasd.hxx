#pragma once

#include <cstdlib>
#include <unordered_set>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "query.hxx"


using google::dense_hash_map;

int separateWords(std::string inputString, std::vector<std::string> &wordVector,const char separator ) {	
	std::string currentword = "";
	
	for (int i = 0; i < inputString.size(); i++) {
		if (inputString[i] == separator) {
			if (currentword.size() == 0 ) {
				continue;
			}
			
			wordVector.push_back(currentword);
			currentword = "";
			continue;
		}
		
		currentword += inputString[i];
			
	}
	
	return 0;
}

std::vector<std::string> string_split(std::string s, const char delimiter)
{
    size_t start=0;
    size_t end=s.find_first_of(delimiter);
    
    std::vector<std::string> output;
    
    while (end <= std::string::npos)
    {
	    output.emplace_back(s.substr(start, end-start));

	    if (end == std::string::npos)
	    	break;

    	start=end+1;
    	end = s.find_first_of(delimiter, start);
    }
    
    return output;
}






class QueryManager {
	private:
		std::string* source;
		int srcSize;
		
		std::vector<TimeQuery> time_queries_;
		std::vector<CountQuery> count_queries_;
		//std::vector<TripleContainer> storebuffer_;
		TripleContainer* storebuffer_;
		//Next position in the buffer; indicates also the length of the buffer.
		int bufferindex_ = 0;
		
		CircularBuffer<unsigned long int> timestamp_pointer_;
		CircularBuffer<TripleContainer> storepointer_;	
    		dense_hash_map<size_t, std::string> resourcemap_;
		
	public:
		QueryManager(std::string* source, int srcSize, int buffSize)   {
			this->srcSize = srcSize;
			this->source = source;
			
			timestamp_pointer_.pointer = (unsigned long int*) malloc(buffSize * sizeof(unsigned long int));
			timestamp_pointer_.size = buffSize;
			
			cudaMalloc(&storepointer_.pointer, buffSize * sizeof(TripleContainer));
			storepointer_.size = buffSize;
			
			storebuffer_ = (TripleContainer*) malloc(buffSize * sizeof(TripleContainer));
			
			resourcemap_.set_empty_key(NULL);      
		}


		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan + storepointer_.pointer, storebuffer_ + hostSpan, copySize * sizeof(TripleContainer), cudaMemcpyHostToDevice); 
		}

		void advancestorepointer() {
			int copySize = bufferindex_;
			
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

			bufferindex_ = 0;
		}
		
		void checkStep() {	
			for (auto &query : count_queries_)  {
				query.incrementCount();
				if (query.isReady()) {
					advancestorepointer();
					query.setWindowEnd(storepointer_.end);
					query.launch();
					query.printResults(resourcemap_);

				}
			}
			
			for (auto &query : time_queries_) {
				if (query.isReady(timestamp_pointer_.pointer[timestamp_pointer_.end - 1])) {
					struct timeval beginQ, endQ, endCpy;
					gettimeofday(&beginQ, NULL);
					
					advancestorepointer();
					gettimeofday(&endCpy, NULL);
					
					query.setWindowEnd(storepointer_.end - 1);							
					query.launch();
					query.printResults(resourcemap_);
					query.setWindowEnd(1);
					
					gettimeofday(&endQ, NULL);
					float QTime = (endQ.tv_sec - beginQ.tv_sec ) * 1000 + ((float) endQ.tv_usec - (float) beginQ.tv_usec) / 1000 ;
					float storetime =  (endCpy.tv_sec - beginQ.tv_sec ) * 1000 + ((float) endCpy.tv_usec - (float) beginQ.tv_usec) / 1000 ;
					
					timeStoreVector.push_back(storetime);
					timeQueryVector.push_back(QTime);  
				}				
			}
		}
		
		void start() {
			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			unsigned long int ts = startingTs.tv_sec * 1000 + startingTs.tv_usec / (1000);

			for (auto &query : time_queries_) {
				query.setStartingTimestamp(ts);
			}
			
			usleep(1);
			float testtime = 0;
			for (int i =0; i <srcSize; i++) {
			
					

				
				TripleContainer currentTriple;
 
 				struct timeval beginTest, endTest;
				gettimeofday(&beginTest, NULL);
 
				std::vector<std::string> triple = string_split(source[i], ' ');
				//separateWords(source[i], triple, ' ');

				

				gettimeofday(&endTest, NULL);	
				testtime +=  (endTest.tv_sec - beginTest.tv_sec ) * 1000 + ((float) endTest.tv_usec - (float) beginTest.tv_usec) / 1000 ;
					
				
				
				currentTriple.subject = h_func(triple[0]);
				currentTriple.predicate = h_func(triple[1]);
				currentTriple.object = h_func(triple[2]);

				resourcemap_[currentTriple.subject] = triple[0];
				resourcemap_[currentTriple.predicate] = triple[1];
  		       		resourcemap_[currentTriple.object] = triple[2] ;
					
				
		
				struct timeval tp;
				gettimeofday(&tp, NULL);
				unsigned long int ms = 0;
				if (triple.size() == 5) {
					ms = std::stol(triple[4]);
					if (i == 0) {
						for (auto &query : time_queries_) {
							query.setStartingTimestamp(ms - 1);
						}
					}
				} else {
					ms = tp.tv_sec * 1000 + tp.tv_usec / (1000);
				}

				timestamp_pointer_.pointer[timestamp_pointer_.end] = ms;
				timestamp_pointer_.end = (timestamp_pointer_.end + 1) % timestamp_pointer_.size;
				timestamp_pointer_.begin = timestamp_pointer_.end;
				
				
				
				
				storebuffer_[bufferindex_] = currentTriple;
				bufferindex_++;
				checkStep();
				
			}
			testVector.push_back(testtime);
			
			//TODO vedere se occorre tenere o no quest'ultima parte
			if (bufferindex_ != 0)  {
				advancestorepointer();	 
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
			}
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
		
		unsigned long int timeParse(std:: string word) {
			char last = word[word.length() -1];
			//U_SEC TIME
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
			unsigned long int window = 0;
			unsigned long int step = 0;
			
			
			bool addAll = false;
									
									
			std::string word = nextWord(&pointer, end, ' ');

			/**JOIN VALUES**/
			std::vector<std::string> variable_stack;
			Operation* currentOp;
			/**************/
			
			if (word == "BASE") {
			 	//TODO IMPLEMENTATION OF BASE
			}
			
			while(word == "PREFIX") {
				std::string prefix = nextWord(&pointer, end, ':');
				std::string prefixUri = nextWord(&pointer, end, ' ');
				prefixes[prefix] = prefixUri;
				//TODO CHECK DUPLICATE PREFIX
				word = nextWord(&pointer, end, ' ');				
			}
			
			
			
			testWord("FROM", word);
			word = nextWord(&pointer, end, ' ');				
			
			//TODO 'NAMED' TOKEN
		
			testWord("STREAM", word);
			
						
			//TODO INDETIFY WHAT TO DO WITH STREAM IRI	
			std::string streamIri = nextWord(&pointer, end, ' ');				

			word = nextWord(&pointer, end, ' ');			
			testWord("RANGE", word);
			
			word = nextWord(&pointer, end, ' ');				
			
			if (word == "TRIPLES") {
				word = nextWord(&pointer, end, ' ');
				window = std::stoul(word);
				logical = false;
				//TODO NEED TO CHECK IF NO INTEGER IS INSERTED
			}
			
			else {
				//TODO VERIFICARE SE RICEVO UN INT SEGUITO DA UNIT° DI MISURA / SPAZIO VALIDO O NO
				logical = true;
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
					
					std::vector<std::string> addedvar;
					std::vector<int> innervar;
					std::vector<int> outervar;
					std::vector<int> joinindex;
					
					for (int i = 0; i <3; i ++ ) {
						word = nextWord(&pointer, end, ' ');
						
						std::cout << "FOUND " << word << std::endl;
						
						if (word[0] == '?') {
							word = word.substr(1);
							if (addAll) {
								if (std::find(variables.begin(), variables.end(), word) == variables.end()) {
									variables.push_back(word);
								} 								
							}
							selectVariable.push_back(word);
							
							bool found = false;
							
							for (int k = 0;  k < variable_stack.size(); k++) {
								std::cout << "word " << word << " CHECK " << k << " value " << variable_stack[k] << std::endl; 
								if (variable_stack[k] == word) {
									innervar.push_back(k);
									outervar.push_back(selectVariable.size() - 1);
									found = true;
									break;
								}
							}
							
							if (!found) {
								joinindex.push_back(selectVariable.size() - 1);
								addedvar.push_back(word);
							}
														
						} else {
							constants.push_back(h_func(word));
							arr += static_cast<int> (stdArr[i]);   
						}
						
					}
					
					
					SelectOperation* currentselect = new SelectOperation(constants, selectVariable, arr);
					selectOperations.push_back(currentselect);
					
					//CREATE JOIN OPERATION
					if (variable_stack.size() == 0) {
						currentOp = currentselect;
						variable_stack.insert(variable_stack.end(), selectVariable.begin(), selectVariable.end());
					} else {
						
						if (addedvar.size() == selectVariable.size()) {
							//TODO VEDERE COSA FARE SE L'ORDINE é SBAGLIATO		
							std::cout << "NOT IMPLEMENTED YET" << std::endl;
							exit(1);	
						} else {
							variable_stack.insert(variable_stack.end(), addedvar.begin(), addedvar.end());
							JoinOperation* joinop = new JoinOperation(currentOp->getResultAddress(), currentselect->getResultAddress(), innervar, outervar, joinindex, variable_stack);
							
							joinOperations.push_back(joinop);
							currentOp = joinop;
						}
					}
					
					
					
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
				std::cout << "WINDOW SIZE IS " << window << " STEP SIZE IS " << step << std::endl;
				TimeQuery query(selectOperations, joinOperations, storepointer_, timestamp_pointer_, variables, window, step);
				time_queries_.push_back(query);
			}
			
			else {
				CountQuery query(selectOperations, joinOperations, storepointer_, variables, window);
				count_queries_.push_back(query);	
			}
			
			
			std::cout << "SELECT SIZE IS " << selectOperations.size() << " JOIN SIZE IS " << joinOperations.size() << std::endl;
				
		}
};




