#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sparsehash/dense_hash_map>
#include <sys/time.h>

using google::dense_hash_map;

//TODO implementare la projection su gpu
//TODO PASSARE I TIMESTAMP DA US A MS (OVERFLOW DI DAY E HOUR)
//TODO 
//VARIABILI PER TESTING, DA RIMUOVERE DAL CODICE FINALE
int VALUE = 0;
std::vector<float> timeCuVector;                
std::vector<long int> timeExVector;
//**END TESTING***//

template <std::size_t FnvPrime, std::size_t OffsetBasis>
struct basic_fnv_1
{
    std::size_t operator()(std::string const& text) const
    {
        std::size_t hash = OffsetBasis;
         for(std::string::const_iterator it = text.begin(), end = text.end();
                 it != end; ++it)
         {
             hash *= FnvPrime;
             hash ^= *it;
         }
         return hash;

     }
};

const basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;

//struct to contains a single triple with int type.
struct tripleContainer {
        size_t subject;
        size_t predicate;
        size_t object;

	void print() {
		std::cout << subject << " " << predicate << " " << object << std::endl;
	}
};

//Struct for circular buffer
template<typename type_t>
struct circularBuffer {
	type_t* pointer;
	int begin;
	int end;
	int size;
	
	circularBuffer() : pointer(0), begin(0), end(0), size(0) {}
	
	int getLength() {
		return (abs(end - begin + size) % size);
	}
	
	void advanceBegin(int step){
		begin = (begin + step) % size;
	}	
};

struct Binding {
	size_t* pointer;
	int width;
	int height;
	std::vector<std::string> header;
	
	Binding() {}
	
	Binding(int width, int height) {
		cudaMalloc(&pointer, width * height *  sizeof(size_t));
		this->width = width;
		this->height = height;
	}
};


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


class Operation {

	protected:
		Binding* result;	
		std::vector<std::string> variables;

	public:
		Binding* getResult() {
			return this->result;
		}
	
		void setResult(Binding* result) {
			this->result = result;
		}
			
		std::vector<std::string> getVariables() {
			return variables;
		}
	
		Binding** getResultAddress() {
			return &result;
		}
};


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

enum class SelectArr { S = 1, P = 2, O = 4, SP = 3, SO = 5, PO = 6, SPO = 7};

__global__ void unarySelect (circularBuffer<tripleContainer> src, int target_pos, int first_pos, int second_pos, size_t value, size_t* dest, int width, int* size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;

	size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};

	if (temp[target_pos] == value) {
		int add = atomicAdd(size, 1);
		size_t* dest_p = (size_t*) (dest + add * width) ;
		*dest_p = temp[first_pos];
		*(dest_p + 1) = temp[second_pos];

	}
}

__global__ void binarySelect (circularBuffer<tripleContainer> src, int target_pos, int target_pos2, int dest_pos, size_t value, size_t value2, size_t* dest, int width, int* size) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
			return;
		}	

		int newIndex = (src.begin + index) % src.size;

		size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};

		if ((temp[target_pos] == value) && (temp[target_pos2] == value2)) {
			int add = atomicAdd(size, 1);
			size_t* dest_p = (size_t*) (dest + add * width) ;
			*dest_p = temp[dest_pos];	
		}
}


class SelectOperation : public Operation
{
	private:
		std::vector<size_t> constants;
		int arr;
		
	public:
		SelectOperation(std::vector<size_t> constants, std::vector<std::string> variables, int arr) {
			this->variables = variables;
			this->constants = constants;	
			this->arr = arr;
		}

		int getArr() {
			return this-> arr;
		}
			
		std::vector<size_t> getQuery() {
			return this->constants;
		}

		void rdfSelect(circularBuffer<tripleContainer> d_pointer, const int storeSize) {
			
			//Initialize elements	
			int* d_resultSize;
			int h_resultSize  = 0;
			cudaMalloc(&d_resultSize, sizeof(int));
			cudaMemcpy(d_resultSize, &h_resultSize, sizeof(int), cudaMemcpyHostToDevice);
	
			//INSERIRE DIVISIONE CORRETTA
			int gridSize = 300;
			int blockSize = (storeSize / gridSize) + 1;
		
			result = new Binding(variables.size(), storeSize);
						
			switch(arr) {

				case(1): {
					unarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 1, 2, constants[0], result->pointer, result->width, d_resultSize);
					break;
				}

				case(2): {
					unarySelect<<<gridSize,blockSize>>>(d_pointer,  1, 0, 2, constants[0], result->pointer, result->width, d_resultSize);
					break;
				}
					
				case(4): {
			        unarySelect<<<gridSize,blockSize>>>(d_pointer,  2, 0, 1, constants[0], result->pointer, result->width, d_resultSize);
			        break;
				}
		
				case(3): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 1, 2, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
				case(5): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 0, 2, 1, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
				case(6): {
					binarySelect<<<gridSize,blockSize>>>(d_pointer, 1, 2, 0, constants[0], constants[1], result->pointer, result->width, d_resultSize);
					break;
				}
					/*
				case(7): {
					cudaMemcpy(result->data(), d_pointer.rdfStore.pointer, storeSize * sizeof(tripleContainer), cudaMemcpyDeviceToDevice);
					cudaMemcpy(d_resultSize, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
			                break;
				}*/
		
			}
	
	
			cudaMemcpy(&h_resultSize, d_resultSize, sizeof(int), cudaMemcpyDeviceToHost);
			
			result->height  =  h_resultSize;
					
			cudaFree(d_resultSize);
		}
	
};



void rdfJoin(Binding* innerTable, Binding* outerTable, std::vector<std::string> joinMask) {
	return;
}



class Query {
	protected:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		circularBuffer<tripleContainer> windowPointer;

	public:
		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, circularBuffer<tripleContainer> rdfPointer) {
			this->join = join;
			this->select = select;
			this->windowPointer = rdfPointer;
		}

		virtual void setWindowEnd(int step) {
			windowPointer.end = step;
		}
		
		/**
		* Function for managing query execution
		**/
		//TODO Verificare se si puo migliorare
		void startQuery() {
			int storeSize =  windowPointer.getLength();			
			
			for (auto op : select) {
				op->rdfSelect(windowPointer, storeSize);
			}


		}

		//TODO modificare quando si sapra come utilizzare i risultati
		void printResults(dense_hash_map<size_t, std::string> mapH) {

			int w = 0;
			for (auto op : select) {
				if (w == 0) VALUE += op->getResult()->height;
				
				std::cout << "PRINTING " << w << " select" << std::endl;
				
				Binding* d_result = op->getResult();
				
				size_t* final_binding = (size_t*) malloc(d_result->height * d_result->width * sizeof(size_t));
				cudaMemcpy(final_binding, d_result->pointer, d_result->width * sizeof(size_t) * d_result->height, cudaMemcpyDeviceToHost);
				
				for (int z = 0; z < d_result->header.size(); z++) {
					std::cout << "header are " << d_result->header[z] << std::endl;
				}
				
				for (int i =0; i < d_result->height; i++) {
					for (int k = 0; k < d_result->width; k++) {
						std::cout << "result is " << mapH[ final_binding[i + k]] << " ";
					}
					
					std::cout << std::endl;
					
				}	
			
				w++;
				cudaFree(d_result->pointer);
			}
	

		/*	for (auto op : join) {
				std::cout << "join result is " << op->getInnerResult()->size() << std::endl;
				std::vector<tripleContainer> innerRes = from_mem(*op->getInnerResult());
				std::vector<const char*> innerHash;
				for (int i =0; i < innerRes.size(); i++) {
					innerHash.push_back( mapH[innerRes[i].subject]);
                                        innerHash.push_back( mapH[innerRes[i].predicate]);
                                        innerHash.push_back( mapH[innerRes[i].object]);
				}
				
				std::vector<tripleContainer> outerRes = from_mem(*op->getOuterResult());
				std::vector<const char*> outerHash;
				for (int i =0; i< outerRes.size(); i++) {
					outerHash.push_back( mapH[outerRes[i].subject]);
                                        outerHash.push_back( mapH[outerRes[i].predicate]);
                                        outerHash.push_back( mapH[outerRes[i].object]);
				}
				VALUE += op->getInnerResult()->size();
				cudaFree(op->getInnerResult()->data());
				cudaFree(op->getOuterResult()->data());
			}*/
					
		}
		
	
};


class CountQuery : public Query {
	private:
		int count;
		int currentCount;

	public:
		CountQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			circularBuffer<tripleContainer> rdfPointer,
			int count) : Query(select, join, rdfPointer) {

				this->count = count;
				this->currentCount = 0;
		}
		
		void incrementCount() {
			this->currentCount++;
		}
		
		bool isReady() {
			return (currentCount == count);
		}
		
		void launch() {
			startQuery();
			windowPointer.advanceBegin(count);
			currentCount = 0;
		}
		
		~CountQuery() {}
};

class TimeQuery : public Query {
	private:
		circularBuffer<long int> timestampPointer;
		//TIME IS IN U_SEC
		long int stepTime;
		long int windowTime;
		long int lastTimestamp;
		
	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			circularBuffer<tripleContainer> rdfPointer, circularBuffer<long int> timestampPointer,
			int windowTime, int stepTime) : Query(select, join, rdfPointer) {
				this->stepTime = stepTime;
				this->windowTime = windowTime;
				
				this->lastTimestamp = 0;
				this->timestampPointer = timestampPointer;
		}
		
		void setWindowEnd(int step)  {
			Query::setWindowEnd(step);
			timestampPointer.end = step;
		}

		bool isReady(long int newTimestamp) {
			return (lastTimestamp + windowTime < newTimestamp);
		}

		void setStartingTimestamp(long int timestamp) {
			this->lastTimestamp = timestamp;
		}

		void launch() {	
			//Update new starting value of buffer
			int newBegin = 0;
			for(int i = timestampPointer.begin; i  != timestampPointer.end; i = (i + 1) % timestampPointer.size) {	
				if (timestampPointer.pointer[i] > lastTimestamp) {
					newBegin = i;
					break;
				}				
			}
							
			windowPointer.begin = newBegin;
			timestampPointer.begin = newBegin;
			
			//Lancuh query and print results
			startQuery();
	
			//Update window timestamp value
			lastTimestamp += stepTime;
		}

		~TimeQuery() {}
};






class QueryManager {
	private:
		int spanTime;
		std::string* source;
		int srcSize;
		
		std::vector<tripleContainer> rdfBuffer;
		std::vector<TimeQuery> timeQueries;
		std::vector<CountQuery> countQueries;
		
		circularBuffer<long int> timestampPointer;
		circularBuffer<tripleContainer> devicePointer;
		
    	dense_hash_map<size_t, std::string> mapH;
		
	public:
		QueryManager(std::string* source, int srcSize, int spanTime,int buffSize)   {
			this->spanTime = spanTime;
			this->srcSize = srcSize;
			this->source = source;
			
			timestampPointer.pointer = (long int*) malloc(buffSize * sizeof(long int));
			timestampPointer.size = buffSize;
			
			cudaMalloc(&devicePointer.pointer, buffSize * sizeof(tripleContainer));
			devicePointer.size = buffSize;
		}
		
		void setDevicePointer(circularBuffer<tripleContainer> devicePointer) {
			this->devicePointer = devicePointer;
		}
		
		circularBuffer<long int> getTimestampPointer() {
			return timestampPointer;
		}
				
		void addTimeQuery(TimeQuery query) {
			timeQueries.push_back(query);
		}
		
		void addCountQuery(CountQuery query) {
			countQueries.push_back(query);
		}

				
		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan + devicePointer.pointer, &rdfBuffer[0] + hostSpan, copySize * sizeof(tripleContainer), cudaMemcpyHostToDevice); 
		}

		
		void advanceDevicePointer() {
			int copySize = rdfBuffer.size();
			
			circularBuffer<tripleContainer> rdfBuff = devicePointer;

			int newEnd = (rdfBuff.end + copySize) % rdfBuff.size;
	 
			if (newEnd < rdfBuff.end) {
				int finalHalf = rdfBuff.size - rdfBuff.end;
				copyElements(devicePointer.end, 0, finalHalf);			
	
				int firstHalf = copySize - finalHalf;
				copyElements(0, finalHalf, firstHalf);			
			} else {
				copyElements(devicePointer.end, 0, copySize);	
			}

			devicePointer.end = newEnd;

			rdfBuffer.clear();
		}
		
		void checkStep() {	
			for (auto &query : countQueries)  {
				query.incrementCount();
				if (query.isReady()) {
					advanceDevicePointer();
					query.setWindowEnd(devicePointer.end);			
					query.launch();
					query.printResults(mapH);
				}
			}
			
			for (auto &query : timeQueries) {
				if (query.isReady(timestampPointer.pointer[timestampPointer.end - 1])) {
					advanceDevicePointer();
					query.setWindowEnd(devicePointer.end - 1);		
					query.launch();
					query.printResults(mapH);
					query.setWindowEnd(1);
				}				
			}
		}
		
		void start() {
			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			long int ts = startingTs.tv_sec * 1000000 + startingTs.tv_usec;

			for (auto &query : timeQueries) {
				query.setStartingTimestamp(ts);
			}
			
			usleep(1);

			basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;

			mapH.set_empty_key(NULL);       
		

			for (int i =0; i <srcSize; i++) {

				
				tripleContainer currentTriple;
 
                                std::vector<std::string> triple;
                                separateWords(source[i], triple, ' ');
			
			        currentTriple.subject = h_func(triple[0]);
                                currentTriple.predicate = h_func(triple[1]);
                                currentTriple.object = h_func(triple[2]);

				mapH[currentTriple.subject] = triple[0];
                                mapH[currentTriple.predicate] = triple[1];
                                mapH[currentTriple.object] = triple[2] ;

				struct timeval tp;
				gettimeofday(&tp, NULL);
				long int ms = tp.tv_sec * 1000000 + tp.tv_usec;


				timestampPointer.pointer[timestampPointer.end] = ms;
				timestampPointer.end = (timestampPointer.end + 1) % timestampPointer.size;
				timestampPointer.begin = timestampPointer.end;
				
				rdfBuffer.push_back(currentTriple);
							
				checkStep();
			//	usleep(spanTime);

			}
			
			//TODO vedere se occorre tenere o no quest'ultima parte
			//***** REMOVE IN FINAL CODE: ONLY FOR TEST (FORSE) *****
			//DA OTTIMIZZARE POICHE VIENE LANCIATO ANCHE QUANDO NON SERVE!!
			advanceDevicePointer();	 
			for (auto &query : timeQueries)  {
				query.setWindowEnd(devicePointer.end);
				query.launch();
				query.printResults(mapH);
			}
			
			for (auto &query :countQueries) {
				query.setWindowEnd(devicePointer.end);
				query.launch();
				query.printResults(mapH);
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
				time = 60 * 1000000 * stoi(word.substr(0, word.length() -1));				
			}
			
			else if (last == 'h') {
				time = 60 * 60 * 1000000 * stoi(word.substr(0, word.length() -1));					
			}
			
			else if (last == 'd') {
				time = 24 * 60 * 60 * 1000000 * stoi(word.substr(0, word.length() -1));		
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
				TimeQuery query(selectOperations, joinOperations, devicePointer, timestampPointer, window, step);
				timeQueries.push_back(query);
			}
			
			else {
				CountQuery query(selectOperations, joinOperations, devicePointer, window);
				countQueries.push_back(query);						
			}				
		}
};





template<typename type_t, typename accuracy>
std::vector<accuracy> stats(std::vector<type_t> input) {
	int size = input.size();
	float mean = 0;
	float variance = 0;
	for (int i = 0; i < size; i++) {
		mean += (accuracy) input[i];
                variance += (accuracy)  (input[i] * input[i]);
        }
        mean = mean / ((accuracy) size);
        variance = variance / ((accuracy) size);
        variance = variance - (mean * mean);
        std::vector<accuracy> statistic;
	statistic.push_back(mean);
	statistic.push_back(variance);
	return statistic;
}



int main(int argc, char** argv) {
 
	using namespace std;
	
	//READ STORE FROM FILE
	ifstream rdfStoreFile (argv[1]);
	string strInput;

	int fileLength = 0;	 
	while (std::getline(rdfStoreFile, strInput)) {
		++fileLength;
	}

	rdfStoreFile.clear();
	rdfStoreFile.seekg(0, ios::beg);

	size_t rdfSize = fileLength  * sizeof(std::string);
	std::string* h_rdfStore = (std::string*) malloc(rdfSize);

	for (int i = 0; i <fileLength; i++) {
		getline(rdfStoreFile,strInput);
	    h_rdfStore[i]  = strInput;
	}
        rdfStoreFile.close();
	//END RDF READ

	struct timeval beginT, endT;
	
	cudaDeviceReset();

        size_t BUFFER_SIZE = 400000;
  
        int N_CYCLE = 100;

	for (int i = 0; i < N_CYCLE; i++) {

		gettimeofday(&beginT, NULL);

		QueryManager manager(h_rdfStore, fileLength, 1, BUFFER_SIZE);
					
		try {
			manager.parseQuery("FROM STREAM <streamUri> RANGE TRIPLES 50000 SELECT ?s WHERE { ?s <http://example.org/int/8> <http://example.org/int/99> .  ?s <http://example.org/int/473> <http://example.org/int/99>   } ");
		}
		catch (std::string exc) {
			std::cout << "Exception raised: " << exc << std::endl;
			exit(1);
		}
		
		manager.start();
		cudaDeviceSynchronize();
		gettimeofday(&endT, NULL);

		float exTime = (endT.tv_sec - beginT.tv_sec ) * 1000 + ((float) endT.tv_usec - (float) beginT.tv_usec) / 1000 ;
					
		timeCuVector.push_back(exTime);

		cout << "Time: " << exTime << endl;
	}

	std::vector<float> statistics = stats<float, float>(timeCuVector);	
        cout << "mean cuda time " << statistics[0] << endl;
        cout << "variance cuda time " << statistics[1] << endl;
	cout << "FINAL VALUE IS " << VALUE << std::endl;
				
        return 0;
}
