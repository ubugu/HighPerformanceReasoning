#pragma once

#include <cstdlib>
#include <unordered_map>
#include <chrono>

#include <sparsehash/dense_hash_map>
    
#include "types.hxx"
#include "parser.hxx"	
#include "query.hxx"


using google::dense_hash_map;


class QueryManager {
	private:
		//THE SOURCE IS A 2D array with width 4 (subject, predicate, object, timestamp)
		std::string* source;
		int srcSize;
		
		std::vector<TimeQuery*> time_queries_;
		std::vector<CountQuery> count_queries_;
		std::vector<size_t> storebuffer_;
		
		CircularBuffer<unsigned long int> timestamp_pointer_;
		CircularBuffer<size_t> storepointer_;	
    		std::unordered_map<size_t, Lit> hashMap_;
    		std::unordered_map<std::string, size_t> inverseHashMap_;
    		
		
	public:
		QueryManager(std::string* source, int srcSize, int buffSize)   {
			this->srcSize = srcSize;
			this->source = source;
			
			timestamp_pointer_.pointer = (unsigned long int*) malloc(buffSize * sizeof(unsigned long int));
			timestamp_pointer_.size = buffSize;
			
			cudaMalloc(&storepointer_.pointer, buffSize * 3 * sizeof(size_t));
			
			storepointer_.size = buffSize;	      
		}

		//Copy elements from host to device
		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan * 3 + storepointer_.pointer, &storebuffer_[0] + hostSpan * 3, copySize * 3 * sizeof(size_t), cudaMemcpyHostToDevice); 
		}

		//Advance circular buffer pointer
		void advancestorepointer() {
			int copySize = storebuffer_.size() / 3;
			
			CircularBuffer<size_t> rdfBuff = storepointer_;

			int newEnd = (rdfBuff.end + copySize) % (rdfBuff.size);
	 		
			if (newEnd < rdfBuff.end) {
				int finalHalf = (rdfBuff.size) - rdfBuff.end;
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
					advancestorepointer();
					query.setWindowEnd(storepointer_.end);
					query.launch();
					query.printResults(hashMap_);
				}
			}
			
			for (auto query : time_queries_) {
				if (query->isReady(timestamp_pointer_.pointer[timestamp_pointer_.end - 1])) {
				
					struct timeval beginQ, endQ, endCpy;
					gettimeofday(&beginQ, NULL);
					
					advancestorepointer();
					gettimeofday(&endCpy, NULL);

					query->setWindowEnd(storepointer_.end - 1);							
					query->launch();
					query->printResults(hashMap_);
					query->setWindowEnd(1);
					
					gettimeofday(&endQ, NULL);
					float QTime = (endQ.tv_sec - beginQ.tv_sec ) * 1000 + ((float) endQ.tv_usec - (float) beginQ.tv_usec) / 1000 ;
					float storetime =  (endCpy.tv_sec - beginQ.tv_sec ) * 1000 + ((float) endCpy.tv_usec - (float) beginQ.tv_usec) / 1000 ;
					
					timeStoreVector.push_back(storetime);
					timeQueryVector.push_back(QTime);  
				}				
			}
		}
		
		//Start streaming
		void start() {
			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			unsigned long int ts = startingTs.tv_sec * 1000 + startingTs.tv_usec / (1000);

			for (auto query : time_queries_) {
				query->setStartingTimestamp(ts);
			}
			
			usleep(1);
		

			auto start = std::chrono::high_resolution_clock::now();		
			for (int i =0; i <srcSize; i++) {
				
				size_t currentTriple;


				for (int k = 0; k < 3; k++) {
									
					//Check if it has been already hashed
					currentTriple = inverseHashMap_[source[i * 4 + k]];
					
					if (currentTriple == 0) {
						//Calculate hash value
						currentTriple = hashFunction(source[i * 4 + k]);
					
						//value 0 is reserved for unbound value
						currentTriple = (currentTriple == 0 ? currentTriple + 1 : currentTriple);
					
						//Check for collisions
						while (hashMap_[currentTriple].stringValue != "") {
							currentTriple++;
						}
						
						hashMap_[currentTriple] = Lit::createLiteral(source[i * 4 + k]);
						inverseHashMap_[source[i * 4 + k]] = currentTriple;					
					}
					
					storebuffer_.push_back(currentTriple);
				}

				
	
				struct timeval tp;
				gettimeofday(&tp, NULL);
				unsigned long int ms = 0;
				
				ms = std::stol(source[i * 4 + 3]);
				if (i == 0) {
					for (auto query : time_queries_) {
						query->setStartingTimestamp(ms - 1);
					}
				}
			
				timestamp_pointer_.pointer[timestamp_pointer_.end] = ms;
				timestamp_pointer_.end = (timestamp_pointer_.end + 1) % timestamp_pointer_.size;
				timestamp_pointer_.begin = timestamp_pointer_.end;
				
				checkStep();

			}
									
			auto finish = std::chrono::high_resolution_clock::now();
			double testtime = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
			testVector.push_back((double) testtime / 1000000);


			//TODO vedere se occorre tenere o no quest'ultima parte
			if (storebuffer_.size() != 0)  {
				advancestorepointer();	 
				for (auto query : time_queries_)  {
					query->setWindowEnd(storepointer_.end);
					query->launch();
					query->printResults(hashMap_);
				}
		
				for (auto &query :count_queries_) {
					query.setWindowEnd(storepointer_.end);
					query.launch();
					query.printResults(hashMap_);
				}
			}
		}
		



		//Function for parsing the query
		void parseQuery(std::string query) {
			
			char* pointer = &query[0];
			char* end = &query[query.length() - 1];
			
			//Prepare elements for query
			std::vector<Operation*> operationsVector;
			CircularBuffer<size_t>* queryRdfPointer = new CircularBuffer<size_t>();
			*queryRdfPointer = storepointer_;
			OutputDirective* outputDirective;
			unsigned long int window = 0;
			unsigned long int step = 0;			
			bool logical = false;	
					
			//Prefixes hash map
			dense_hash_map<std::string, std::string> prefixes;
			prefixes.set_empty_key("");    	
			
			//Start partsing										
			std::string word = nextWord(&pointer, end, ' ');


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
				logical = true;
				window = timeParse(word);
				
				word = nextWord(&pointer, end, ' ');							
				//NEED TO CHECK INTEGER
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
			
			//Code for SELECT directivehttp://www.spaziogames.it/
			if (word == "SELECT")  {
			
				word = nextWord(&pointer, end, ' ');
					
				//Check if there is a projection
				if (word == "*") {
					outputDirective = new SelectDirective();
					
					word = nextWord(&pointer, end, ' ');
					testWord("WHERE", word);
					word = nextWord(&pointer, end, ' ');
					testWord("{", word);
										 
					//Parse operations
					operationsVector =  blockElement(&pointer, end, queryRdfPointer, &hashMap_, &inverseHashMap_); 
					
				} else {
					std::vector<std::string> variables;
					
					//Add projection variables			
					while(word[0] == '?') {					
						variables.push_back(word);
						word = nextWord(&pointer, end, ' ');					
					}
				
					if (variables.size() == 0) {
						std::cout << "Error, no variable inserted" << std::endl;
						exit(-1);
					}	
					
					testWord("WHERE", word);
					word = nextWord(&pointer, end, ' ');
					testWord("{", word);
					
					//Parse operations
					operationsVector =  blockElement(&pointer, end, queryRdfPointer, &hashMap_, &inverseHashMap_); 
				
					//Calculate index of prijection variables
					std::vector<int> variablesIndex; 
					std::vector<std::string> opVariables = operationsVector.back()->getVariables();
					for(int i = 0; i < variables.size(); i++) {
						for (int k =0; k < opVariables.size(); k++) {
							if (variables[i] == opVariables[k]) {
								variablesIndex.push_back(k);
							}
						}
					}
				
					outputDirective = new SelectDirective(variablesIndex);
				}
				
			//Code for ASK directive	
			} else if (word == "ASK") {
				//TODO
				
			//Code for CONSTRUCT directive
			} else if (word == "CONSTRUCT") {
				//TODO
				
			//Throw error for other cases	
			} else {
				throw err("SELECT or ASK or CONSTRUCT", word);
			}
			
			//Check if other token are inserted	
			pointer = nextChar(pointer, end);
			if (pointer != end)  {
				throw std::string("Unexpected token. Expected EOL and found");
			}
			
			//Create the query
			if (logical == true) {
				TimeQuery* query = new TimeQuery(operationsVector, queryRdfPointer, timestamp_pointer_, outputDirective, window, step);
				time_queries_.push_back(query);
				
			} else {
			//	CountQuery query(selectOperations, joinOperations, storepointer_, variables, window);
			//	count_queries_.push_back(query);	
			}

				
		}
};




