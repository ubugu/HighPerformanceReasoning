#pragma once

#include <cstdlib>
#include <unordered_map>
    
#include "types.hxx"
#include "parser.hxx"	
#include "query.hxx"


class QueryManager {
	private:
		/*
		* The source is a linear array which lenght is a multiple of 4.
		* The value are inserted in the order (subject, predicate, object, timestamp).
		*/
		std::string* source;
		int srcSize;
		
		std::vector<TimeQuery*> timeQueries_;
		std::vector<CountQuery*> countQueries_;
		std::vector<size_t> storebuffer_;
		
		CircularBuffer<unsigned long int> timestamp_pointer_;
		CircularBuffer<size_t> storepointer_;	
    		std::unordered_map<size_t, Lit> hashMap_;
    		std::unordered_map<std::string, size_t> inverseHashMap_;
    		
		
	public:
		//Initialize the manager, specifying the source and the circular buffer size.
		QueryManager(std::string* source, int srcSize, int buffSize)   {
			this->srcSize = srcSize;
			this->source = source;
			
			timestamp_pointer_.pointer = (unsigned long int*) malloc(buffSize * sizeof(unsigned long int));
			timestamp_pointer_.size = buffSize;
			
			cudaMalloc(&storepointer_.pointer, buffSize * 3 * sizeof(size_t));
			
			storepointer_.size = buffSize;	      
		}
		
		~QueryManager() {
			cudaFree(storepointer_.pointer);
		}

		//Copy elements from host to device
		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan * 3 + storepointer_.pointer, &storebuffer_[0] + hostSpan * 3, copySize * 3 * sizeof(size_t), cudaMemcpyHostToDevice); 
		}

		//Advance circular buffer pointers
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
		
		//Check if a query is ready
		void checkStep() {	
			for (auto query : countQueries_)  {
				query->incrementCount();
				if (query->isReady()) {
					advancestorepointer();
					query->setWindowEnd(storepointer_.end);
					query->launch();
					query->printResults(hashMap_);
				}
			}
			
			for (auto query : timeQueries_) {
				if (query->isReady(timestamp_pointer_.pointer[timestamp_pointer_.end - 1])) {
				
					advancestorepointer();
					query->setWindowEnd(storepointer_.end - 1);							
					query->launch();
					
					query->printResults(hashMap_);
					query->setWindowEnd(1);

				}				
			}
		}
		
		//Start streaming
		void start() {
			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			unsigned long int ts = startingTs.tv_sec * 1000 + startingTs.tv_usec / (1000);

			for (auto query : timeQueries_) {
				query->setStartingTimestamp(ts);
			}
			
			usleep(1);
			
			int i = 0;
			
			for (; i <srcSize; i++) {
				
				size_t currentTriple;

				for (int k = 0; k < 3; k++) {


					//NORMAL HASH POLICY
					//Calculate hash value
					currentTriple = hashFunction(source[i * 4 + k].c_str(), source[i * 4 + k].size());
					
					//value 0 is reserved for unbound value
					currentTriple = (currentTriple == 0 ? currentTriple + 1 : currentTriple);
					
					//Insert the value into the hash map
					auto mapValue = hashMap_[currentTriple].stringValue;
					if (mapValue != "") {
						if(mapValue != source[i * 4 + k]) {
							//Check for collisions
							do {
								currentTriple++;
							} while (hashMap_[currentTriple].stringValue != "");
						
						} 
					
					} else {
						hashMap_[currentTriple] = Lit::createLiteral(source[i * 4 + k]);
					}

					
					storebuffer_.push_back(currentTriple);
				}

				//Add the triple timestamp
				unsigned long int ms = 0;
				ms = std::stol(source[i * 4 + 3]);
				if (i == 0) {
					for (auto query : timeQueries_) {
						query->setStartingTimestamp(ms - 1);
					}
				}
				timestamp_pointer_.pointer[timestamp_pointer_.end] = ms;
				timestamp_pointer_.end = (timestamp_pointer_.end + 1) % timestamp_pointer_.size;
				timestamp_pointer_.begin = timestamp_pointer_.end;
				
				checkStep();

			}

			//If the stream ended before starting any query, executes the query on all the remaning elements
			//TODO Check if keep or not this part
			if (storebuffer_.size() != 0)  {
				advancestorepointer();
				for (auto query : timeQueries_)  {
					query->setWindowEnd(storepointer_.end);
					query->launch();
					query->printResults(hashMap_);
				}
		
				for (auto query :countQueries_) {
					query->setWindowEnd(storepointer_.end);
					query->launch();
					query->printResults(hashMap_);
				}
			}
			
						
		}
		
		
		




		//Function for starting query parsing
		void parseQuery(std::string query) {
			
			//Initialize pointers to the query string
			char* stringPointer = &query[0];
			char* end = &query[query.length() - 1];
			
			//Prepare elements for query
			std::vector<Operation*> operationsVector; 					//Vector of the basic operations composing the query
			CircularBuffer<size_t>* queryRdfPointer = new CircularBuffer<size_t>();		//Pointer to the circular buffer
			*queryRdfPointer = storepointer_;						
			OutputDirective* outputDirective;						//Output directive (SELECT, CONSTRUCT, ASK)
			std::vector<std::string> projectionVariables;					//Variables for final projection
			unsigned long int window = 0;							
			unsigned long int step = 0;							
			bool logical = false;								
			std::unordered_map<std::string, std::string>prefixes;				//Hash map for managing the prefixes
  	
			
			//Start partsing										
			std::string word = nextWord(&stringPointer, end, ' ');
				
			//Insert prefixes into the hash map
			while(word == "PREFIX") {
				std::string prefix = nextWord(&stringPointer, end, ':');
				std::string prefixUri = nextWord(&stringPointer, end, ' ');
				prefixes[prefix] = prefixUri;	
				
				word = nextWord(&stringPointer, end, ' '); 	
			}

			//Code for SELECT directive
			if (word == "SELECT")  {
				word = nextWord(&stringPointer, end, ' ');
					
				//Check if there is the star operator
				if (word == "*") {
					outputDirective = new SelectDirective();
					
				} else {
					//Add SELECT variables for final projection		
					while(word[0] == '?') {					
						projectionVariables.push_back(word);
						word = nextWord(&stringPointer, end, ' ');					
					}
				
					if (projectionVariables.size() == 0) {
						std::cout << "Error, no variable inserted" << std::endl;
						exit(-1);
					}	
				}
			
			//Throw error if no select found
			} else {
				throw err("SELECT", word);
			}
			
			
			//Read input stream
			word = nextWord(&stringPointer, end, ' ');	
			testWord("FROM", word);

			//TODO INDETIFY WHAT TO DO WITH STREAM IRI
			word = nextWord(&stringPointer, end, ' ');					
			testWord("STREAM", word);					
			std::string streamIri = nextWord(&stringPointer, end, ' ');				

			//Check for window types and values
			word = nextWord(&stringPointer, end, ' ');			
			testWord("RANGE", word);
			
			//TODO check if window/step values are integer
			word = nextWord(&stringPointer, end, ' ');				
			if (word == "TRIPLES") {
				word = nextWord(&stringPointer, end, ' ');
				window = std::stoul(word);
				logical = false;
				
			} else {
				logical = true;
				window = timeParse(word);
				word = nextWord(&stringPointer, end, ' ');							
				if (word == "STEP") {
					word = nextWord(&stringPointer, end, ' ');							
					step = timeParse(word);
					
				} else if (word == "TUMBLING") {
					step = window;
					
				} else {
					throw err("STEP TIME", word);
				}	
			}
			
			//Parse where clause
			word = nextWord(&stringPointer, end, ' ');
			testWord("WHERE", word);
			word = nextWord(&stringPointer, end, ' ');
			testWord("{", word);
								 
			//Parse operations
			operationsVector =  blockElement(&stringPointer, end, queryRdfPointer, &hashMap_, &inverseHashMap_,  prefixes); 
			
			//Calculate index of prijection variables  if the star operator isn't used
			if (projectionVariables.size() != 0) {
				std::vector<int> variablesIndex; 
				std::vector<std::string> opVariables = operationsVector.back()->getVariables();
				for(int i = 0; i < projectionVariables.size(); i++) {
					for (int k =0; k < opVariables.size(); k++) {
						if (projectionVariables[i] == opVariables[k]) {
							variablesIndex.push_back(k);
						}
					}
				}
				outputDirective = new SelectDirective(variablesIndex);
			}
		

		
			//Check if other token are inserted	
			stringPointer = nextChar(stringPointer, end);
			if (stringPointer != end)  {
				throw std::string("Unexpected token. Expected EOL and found");
			}
			
			//Create the query
			if (logical == true) {
				TimeQuery* query = new TimeQuery(operationsVector, queryRdfPointer, timestamp_pointer_, outputDirective, window, step);
				timeQueries_.push_back(query);
				
			} else {
				CountQuery* query = new CountQuery(operationsVector, queryRdfPointer, outputDirective, window);
				countQueries_.push_back(query);	
			}

				
		}
};




