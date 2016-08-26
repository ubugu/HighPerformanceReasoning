#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sparsehash/dense_hash_map>
#include <sys/time.h>


using namespace mgpu;
using google::dense_hash_map;

//TODO implementare la projection su gpu
//TODO PASSARE I TIMESTAMP DA US A MS (OVERFLOW DI DAY E HOUR)
//TODO 
//VARIABILI PER TESTING, DA RIMUOVERE DAL CODICE FINALE
int VALUE = 0;
std::vector<float> timeCuVector;                
std::vector<long int> timeExVector;
bool isLaunched = false;
//**END TESTING***//


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
	
	Binding() {
	
	}
	
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
		
		std::vector<std::string> joinMask;

	public:
		JoinOperation(Binding** innerTable, Binding** outerTable, std::string joinMask) {
			this->innerTable = innerTable;
			this->outerTable = outerTable;
			separateWords(joinMask, this->joinMask, ' ');
		};
			
		Binding* getInnerTable() {
			return *this->innerTable;
		};
		
		Binding* getOuterTable() {
			return *this->outerTable;
		};
		
		std::vector<std::string> getJoinMask() {
			return this->joinMask;
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

__global__ void binarySelect (circularBuffer<tripleContainer> src, int target_pos, int target_pos2, int dest_pos, size_t* value, size_t* value2, size_t* dest, int width, int* size) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
			return;
		}	

		int newIndex = (src.begin + index) % src.size;

		size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};

		if ((temp[target_pos] == (*value)) && (temp[target_pos2] == (*value2))) {
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
		};

		int getArr() {
			return this-> arr;
		}
			
		std::vector<size_t> getQuery() {
			return this->constants;
		};

		/*
		* Make multiple select query, with specified comparison condition,
		* on a triple store. Both queries and the store are supposed to 
		* be already on the device. 
		* 
		* @param d_selectQueries : the array in which are saved the select values
		* @param d_storePointer : pointer on the device to the triple store
		* @param storeSize : size of the triple store
		* @return a vector of type mem_t in which are saved the query results.
		*/
		void rdfSelect(circularBuffer<tripleContainer> d_pointer, const int storeSize) 
		{	
			//Initialize elements	
			int* d_resultSize;
			cudaMalloc(&d_resultSize, sizeof(int));
			int h_resultSize  = 0;

			cudaMemcpy(d_resultSize, &h_resultSize, sizeof(int), cudaMemcpyHostToDevice);
	
			//INSERIRE DIVISIONE CORRETTA
			int gridSize = 300;
			int blockSize = (storeSize / gridSize) + 1;
		
			result = new Binding(2, storeSize);
						
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
		
		/*		case(3): {
					size_t* value1 = &(query->subject);
					size_t* value2 = &(query->predicate);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.predicate, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}
				case(5): {
					size_t* value1 = &(query->subject);
					size_t* value2 = &(query->object);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.object, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}
				case(6): {
					size_t* value1 = &(query->predicate);
					size_t* value2 = &(query->object);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, d_pointer.object, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}
					
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


//Sorter for sorting the triple due to theorder defined by the sortMask
 class TripleSorter {
	private:
		int sortMask[3];
	public:
		TripleSorter(JoinMask sortMask[3]) {
			this->sortMask[0] = static_cast<int> (sortMask[0]);
			this->sortMask[1] = static_cast<int> (sortMask[1]);
			this->sortMask[2] = static_cast<int> (sortMask[2]);
				
		}
		
		MGPU_DEVICE bool operator() (tripleContainer a, tripleContainer b) {
			size_t tripleA[3] = {a.subject, a.predicate, a.object};
			size_t tripleB[3] = {b.subject, b.predicate, b.object};

							
			if ((sortMask[0] != -1) && (tripleA[sortMask[0]] < tripleB[sortMask[0]])) {
				return true;
			}
			
			if ((sortMask[1] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] < tripleB[sortMask[1]])) {
				return true;
			}
			
			if ((sortMask[2] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] == tripleB[sortMask[1]]) && (tripleA[sortMask[2]] < tripleB[sortMask[2]])) {
				return true;
			}
			
			return false;
		}
};


//Sorter for sorting the triple due to theorder defined by the sortMask
 class Sorter {
	private:
		int sortMask[3];
	public:
		Sorter(int sortMask[3]) {
			this->sortMask[0] =  (sortMask[0]);
			this->sortMask[1] =  (sortMask[1]);
			this->sortMask[2] =  (sortMask[2]);
				
		}
		
		
		MGPU_DEVICE bool operator() (size_t a, size_t b) {
		
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			        
			printf("pointers are %p , %p \n", &a, &b);
		
			/*size_t tripleA[3] = {a.subject, a.predicate, a.object};
			size_t tripleB[3] = {b.subject, b.predicate, b.object};*/

				/*			
			if ((sortMask[0] != -1) && (tripleA[sortMask[0]] < tripleB[sortMask[0]])) {
				return true;
			}
			
			if ((sortMask[1] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] < tripleB[sortMask[1]])) {
				return true;
			}
			
			if ((sortMask[2] != -1) && (tripleA[sortMask[0]] == tripleB[sortMask[0]]) && (tripleA[sortMask[1]] == tripleB[sortMask[1]]) && (tripleA[sortMask[2]] < tripleB[sortMask[2]])) {
				return true;
			}*/
			
			return false;
		}
};


struct mask_t {
	int subject;
	int predicate;
	int object;
};
 

__global__ void reorderTriple(tripleContainer* src, tripleContainer* dest, int maxSize, mask_t mask) {
		
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}

	size_t triple[3] = {src[destIndex].subject, src[destIndex].predicate, src[destIndex].object};

	dest[destIndex] = {triple[mask.subject], triple[mask.predicate], triple[mask.object]};
}

__global__ void indexCopy(tripleContainer* innerSrc, tripleContainer* innerDest, tripleContainer* outerSrc, tripleContainer* outerDest, int2* srcIndex, int maxSize)  {

	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}
	
	int innerIndex = srcIndex[destIndex].x;
	int outerIndex = srcIndex[destIndex].y;
	
	innerDest[destIndex] = innerSrc[innerIndex];	
	outerDest[destIndex] = outerSrc[outerIndex];
}

std::vector<mem_t<tripleContainer>*> rdfJoin(Binding* innerTable, Binding* outerTable, std::vector<std::string> joinMask) {
	//TODO Migliorare la join nel reordering delle triple
	standard_context_t context;
	std::vector<mem_t<tripleContainer>*> finalResults;
	
		
	
/*
	
	mask_t mask;
	mask.subject = static_cast<int> (outerMask[0]);
	mask.predicate = static_cast<int> (outerMask[1]);
	mask.object = static_cast<int> (outerMask[2]);	
	
	//TODO SETTARE DIVISIONE
	int gridSize = 124;
	int blockSize = (outerSize/ gridSize) + 1;
	mem_t<tripleContainer>* tempOuter = new mem_t<tripleContainer>(outerSize, context);
	reorderTriple<<<gridSize, blockSize>>>(outerTable, tempOuter->data(), outerSize, mask);
	
	std::cout << "LAUNCH MEGER " << std::endl;
	//Sort the two input array
	mergesort<launch_params_t<128, 2>>(d_iter, innerTable->height, *innerSorter, context);
	exit(1);
//	mergesort<launch_params_t<128, 2>>(outerTable->pointer, outerSize , *innerSorter, context);
/*	
	mem_t<int2> joinResult = inner_join<launch_params_t<128,2>>( innerTable, innerSize, tempOuter->data(), outerSize,  *innerSorter, context);
		
	std::cout << "JOIN RESULT SIZE IS " << joinResult.size() << std::endl;
	
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
        mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//TODO BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	gridSize = 64;
	blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), tempOuter->data(), outerResults->data(), joinResult.data(), joinResult.size());
	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);
	cudaFree(tempOuter->data());
	free(tempOuter);
	*/
	return finalResults;
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

		
			/*
			
	
			for (int i = 0; i < join.size(); i++) {
				Binding* innerTable = join[i]->getInnerTable();
				Binding* outerTable = join[i]->getOuterTable();
				std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin(innerTable, outerTable, join[i]->getJoinMask());
			/*	join[i]->setInnerResult(joinResult[0]);
				join[i]->setOuterResult(joinResult[1]);			
			}*/
			
	
		}

		//TODO modificare quando si sapra come utilizzare i risultati
		void printResults(dense_hash_map<size_t, std::string> mapH) {

			int w = 0;
			for (auto op : select) {
				printf("W VALUE IS %i \n", w);
				if (w == 0) VALUE += op->getResult()->height;
				
				Binding* d_result = op->getResult();
				
				size_t* final_binding = (size_t*) malloc(d_result->height * d_result->width * sizeof(size_t));
				cudaMemcpy(final_binding, d_result->pointer, d_result->width * sizeof(size_t) * d_result->height, cudaMemcpyDeviceToHost);
				std::cout << "size is " << d_result->height << std::endl;
				std::cout << "width is " << d_result->width << std::endl;
				
				for (int z = 0; z < d_result->header.size(); z++) {
					std::cout << "header are " << d_result->header[z] << std::endl;
				}
				for (int i =0; i < d_result->height; i++) {
					//for (int k = 0; k < d_result->width; k++) {
						//size_t current = final_binding[i + k];
						std::cout << "result is " << mapH[ final_binding[i]] << " " <<  mapH[final_binding[i + 1]] << std::endl;
					//}
					
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


dense_hash_map<size_t, std::string> mapH;


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
    //            dense_hash_map<size_t, std::string> mapH;


	public:
		QueryManager(std::string* source, int srcSize, int spanTime,int buffSize)   {
			this->spanTime = spanTime;
			this->srcSize = srcSize;
			this->source = source;
			
			long int* timestamp = (long int*) malloc(buffSize * sizeof(long int));
			timestampPointer.pointer = timestamp;
			timestampPointer.size = buffSize;
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

		//	mapH.set_empty_key(NULL);       
		

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
					
					int arr;
					std::vector<std::size_t> constants;
					std::vector<std::string> selectVariable;
					
					const SelectArr stdArr[3] = {SelectArr::S, SelectArr::P, SelectArr::O};
					
					
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
					
					/*
					dense_hash_map<Operation**, std::vector<std::string>> currentStack;
					stackOperations.set_empty_key(NULL);
				
					Operation* op = currentOp;
					for (auto var : selectVaraible) {
						
						if (stackVaraible[var] == NULL) {
							stackOperations[var] = &op;
						} 
						else {
							currentStack.[stackVaraible[var]].push_back(var);
						}						
					}

					//COME CICLARE SUGLI ELEMENTI DELL' HASH MAP
					for (auto oper : currentStack) {
						auto operVar = currentStack[oper];
						
						oper = newnewOp;
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
		
		struct timeval beginPr, beginCu, beginEx, end;
		
		gettimeofday(&beginPr, NULL);	
		cudaDeviceReset();
		standard_context_t context;

        size_t BUFFER_SIZE = 400000;
		circularBuffer<tripleContainer> windowPointer;




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


                //read store from rdfStore
                for (int i = 0; i <fileLength; i++) {
			getline(rdfStoreFile,strInput);
                        h_rdfStore[i]  = strInput;
                }

                rdfStoreFile.close();
		mapH.set_empty_key(NULL);    
                int N_CYCLE = 1;
		for (int i = 0; i < N_CYCLE; i++) {

			gettimeofday(&beginCu, NULL);
			QueryManager manager(h_rdfStore, fileLength, 1, BUFFER_SIZE);
			cudaMalloc(&windowPointer.pointer, BUFFER_SIZE * sizeof(tripleContainer));
					
			int begin = 0;
			windowPointer.begin = begin;
			windowPointer.end = begin;
			windowPointer.size = BUFFER_SIZE;	
			manager.setDevicePointer(windowPointer);
			
			try {
				manager.parseQuery("FROM STREAM <streamUri> RANGE TRIPLES 50000 SELECT ?s ?p WHERE { ?s ?p <http://example.org/int/99>  } ");
			}
			catch (std::string exc) {
				std::cout << "Exception raised: " << exc << std::endl;
				exit(1);
			}
			
			
			/*
			std::vector<size_t> constOne;
			constOne.push_back( h_func("<http://example.org/int/99>"));
				
			std::vector<std::string> variables;
			variables.push_back("?s");
			variables.push_back("?p");
		
			SelectOperation  selectOp1(constOne, variables,  static_cast<int> (SelectArr::O));		
			std::vector<SelectOperation*> selectOperations;
			std::vector<JoinOperation*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			
			CountQuery count2(selectOperations, joinOperations, windowPointer, 50000);
			manager.addCountQuery(count2);
			*/
			gettimeofday(&beginEx, NULL);	


			manager.start();


			
			cudaDeviceSynchronize();
			gettimeofday(&end, NULL);



			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			

			
			timeCuVector.push_back(cuTime);

						

			cout << "Cuda time: " << cuTime << endl;


			cudaFree(windowPointer.pointer);
			mapH.clear();
		}

	
		std::vector<float> statistics;
		
		statistics = stats<float, float>(timeCuVector);	
                cout << "mean cuda time " << statistics[0] << endl;
                cout << "variance cuda time " << statistics[1] << endl;

              /*  statistics = stats<long int, double>(timeExVector);
                cout << "mean ex time " << statistics[0] << endl;
                cout << "variance ex time " << statistics[1] << endl;*/


		cout << "FINAL VALUE IS " << VALUE << std::endl;;
		
		long int sum = 0;
		
		for (int i = 0; i < timeCuVector.size(); i++) {
			std::cout<< "time are " << timeCuVector[i] << std::endl;
		}
		

		
                return 0;
}
