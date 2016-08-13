#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>

using namespace mgpu;

int TEST_VALUE[2]  {0, 0};

//struct to contains a single triple with int type.
struct tripleContainer {
        int subject;
        int predicate;
        int object;
};

//Struct for circular buffer
template<typename type_t>
struct circularBuffer {
	type_t* pointer;
	int begin;
	int end;
	int size;
	
	circularBuffer() : pointer(0), begin(0), end(0), size(0) {}
};

//Struct for containing the pointer to an rdf store (divided into subject predicate and object)
template<typename rdf_t, typename arr_t>
struct triplePointer {
	rdf_t rdfStore;
	arr_t subject;
	arr_t predicate;
	arr_t object;
};

/*
* Specific implementation of triplePointer for ciruclar buffer.
* Offers methods for managing the attributes of the class.
*/
struct deviceCircularBuffer : triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> {
	void setValues(int begin, int end, int size) {
		setBegin(begin);
		setEnd(end);
		setSize(size);
	}	

	void setBegin(int begin) {
		rdfStore.begin = begin;
		subject.begin = begin;
		predicate.begin = begin;
		object.begin = begin;
	}
	
	void setEnd(int end) {
		rdfStore.end = end;
		subject.end = end;
		predicate.end = end;
		object.end = end;
	}
	
	void setSize(int size) {
		rdfStore.size = size;
		subject.size = size;
		predicate.size = size;
		object.size = size;
	}
	
	void advanceBegin(int step){
		int newBegin = (rdfStore.begin + step) % rdfStore.size;
		setBegin(newBegin);
	}
	
	void advanceEnd(int step){
		int newEnd = (rdfStore.end + step) % rdfStore.size;
		setEnd(newEnd);
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

std::vector<mem_t<tripleContainer>*> rdfJoin(tripleContainer* innerTable, int innerSize, tripleContainer* outerTable, int outerSize, JoinMask innerMask[3], JoinMask outerMask[3]);

std::vector<mem_t<tripleContainer>*> rdfSelect(const std::vector<tripleContainer*> d_selectQueries, 
		deviceCircularBuffer d_pointer,
		const int storeSize, 
		std::vector<int>  arrs);

class Query {
	public:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		deviceCircularBuffer windowPointer;
		long int lastTimestamp;

		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, deviceCircularBuffer rdfPointer) {
			this->join = join;
			this->select = select;
			this->windowPointer = rdfPointer;
		}

		virtual void advancePointer(int step) {
			windowPointer.advanceEnd(step);
		}
		
		void printResults() {
			int i = 0;
			for (auto op : select) {
				std::vector<tripleContainer> selectResults = from_mem(*(op->getResult()));
				std::cout <<"selct size " << selectResults.size() << std::endl;
				
				cudaFree(op->getResult()->data());
				
				if (i <= 1) {
					TEST_VALUE[i] += selectResults.size();
				}
				
				i++;
			}
			
			for (auto op : join) {
				cudaFree(op->getInnerResult()->data());
				cudaFree(op->getOuterResult()->data());
			}
					
		}
		
		void setStartingTimestamp(long int timestamp) {
			this->lastTimestamp = timestamp;
		}

		
		virtual void launch() =0;
		virtual bool isReady() =0;
		
		/**
		* Function for managing query execution
		**/
		void startQuery() {
			int storeSize =  (abs(windowPointer.rdfStore.end - windowPointer.rdfStore.begin +  windowPointer.rdfStore.size) % windowPointer.rdfStore.size);			
			std::vector<tripleContainer*> d_selectQueries;
			std::vector<int> arrs;
	
			for (int i = 0; i < select.size(); i++) {
				d_selectQueries.push_back(select[i]->getQuery()->data());
				arrs.push_back(select[i]->getArr());
			}
	
			std::vector<mem_t<tripleContainer>*> selectResults = rdfSelect(d_selectQueries, windowPointer, storeSize, arrs);

			for (int i = 0; i < selectResults.size(); i++) {
				select[i]->setResult(selectResults[i]);
			}
	
	
			for (int i = 0; i < join.size(); i++) {
				mem_t<tripleContainer>* innerTable = *join[i]->getInnerTable();
				mem_t<tripleContainer>* outerTable = *join[i]->getOuterTable();
				std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin(innerTable->data(), innerTable->size(), outerTable->data(), outerTable->size(), join[i]->getInnerMask(), join[i]->getOuterMask());
				join[i]->setInnerResult(joinResult[0]);
				join[i]->setOuterResult(joinResult[1]);				
			}
	
		}

};


class CountQuery : public Query {
	private:
		int count;
		int currentCount;

	public:
		CountQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			deviceCircularBuffer rdfPointer,
			int count) : Query(select, join, rdfPointer) {

				this->count = count;
				this->currentCount = 0;
		}
		
		bool isReady() {
			this->currentCount++;
			if (currentCount == count) {
				currentCount = 0;
				return true;
			}
			return false;
		}
		

	
		void launch() {
			startQuery();
			windowPointer.advanceBegin(count);
			printResults();
		}
		
		~CountQuery() {}
};

class TimeQuery : public Query {
	private:
		circularBuffer<long int> timestampPointer;
		long int stepTime;
		long int windowTime;

	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			deviceCircularBuffer rdfPointer, circularBuffer<long int> timestampPointer,
			int windowTime, int stepTime) : Query(select, join, rdfPointer) {
				this->stepTime = stepTime;
				this->windowTime = windowTime;
				this->lastTimestamp = 0;
				this->timestampPointer = timestampPointer;
		}
		

		void advancePointer(int step)  {
			Query::advancePointer(step);
			timestampPointer.end = (timestampPointer.end + step) % timestampPointer.size;
		}

		bool isReady() {
			long int newTimestamp = timestampPointer.pointer[timestampPointer.end -1 ];
	
			if (lastTimestamp + windowTime <= newTimestamp) {
				return true;
			}

			return false;
		}

		void launch() {
			
			int newBegin = 0;
			
			for(int i = timestampPointer.begin; i  != timestampPointer.end; i = (i + 1) % timestampPointer.size) {
				if (timestampPointer.pointer[i] > lastTimestamp) {
					newBegin = i;
					break;
				}				
			}				

			windowPointer.setBegin(newBegin);
			timestampPointer.begin = newBegin;
	
			windowPointer.setEnd(windowPointer.rdfStore.end -1);

			startQuery();
			printResults();
			
			windowPointer.setEnd(windowPointer.rdfStore.end + 1);
			
			lastTimestamp += stepTime;
		}

		~TimeQuery() {}
};




class TripleGenerator {
	private:
		int spanTime;
		tripleContainer* source;
		int srcSize;
		
		std::vector<tripleContainer> rdfBuffer;
		std::vector<int> subjectBuffer;
		std::vector<int> predicateBuffer;
		std::vector<int> objectBuffer;

		std::vector<Query*> queries;
		std::vector<TimeQuery> timeQueries;
		std:.vector<CountQuery> countQueries;
		
		circularBuffer<long int> timestampPointer;
		deviceCircularBuffer devicePointer;

	public:
		TripleGenerator(tripleContainer* source, int srcSize, int spanTime,int buffSize)   {
			this->spanTime = spanTime;
			this->srcSize = srcSize;
			this->source = source;
			long int* timestamp = (long int*) malloc(buffSize * sizeof(long int));
			timestampPointer.pointer = timestamp;
			timestampPointer.size = buffSize;
		}
		
		void setDevicePointer(deviceCircularBuffer devicePointer) {
			this->devicePointer = devicePointer;
		}
		
		circularBuffer<long int> getTimestampPointer() {
			return timestampPointer;
		}
		
		void addQuery(Query &query) {
			queries.push_back(&query);
		}
		
		void addTimeQuery(TimeQuery query) {
			timeQueries.push_back(query);
		}
		
		void addCountQuery(CountQuery query) {
			countQueries.push_back(query);
		}

		
		
		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan + devicePointer.rdfStore.pointer, &rdfBuffer[0] + hostSpan, copySize * sizeof(tripleContainer), cudaMemcpyHostToDevice); 
			cudaMemcpy(deviceSpan + devicePointer.subject.pointer, &subjectBuffer[0] + hostSpan, copySize * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceSpan + devicePointer.predicate.pointer,&predicateBuffer[0] + hostSpan, copySize * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceSpan + devicePointer.object.pointer, &objectBuffer[0] + hostSpan, copySize * sizeof(int), cudaMemcpyHostToDevice);
		}

		
		void advanceDevicePointer() {
			int copySize = rdfBuffer.size();
			
			circularBuffer<tripleContainer> rdfBuff = devicePointer.rdfStore;

			int newEnd = (rdfBuff.end + copySize) % rdfBuff.size;

				 
			if (newEnd < rdfBuff.end) {
				int finalHalf = rdfBuff.size - rdfBuff.end;
				copyElements(devicePointer.rdfStore.end, 0, finalHalf);			
	
				int firstHalf = copySize - finalHalf;
				copyElements(0, finalHalf, firstHalf);			
			} else {
				copyElements(devicePointer.rdfStore.end, 0, copySize);	
			}
			


			devicePointer.setEnd(newEnd);

			rdfBuffer.clear();
			subjectBuffer.clear();
			predicateBuffer.clear();
			objectBuffer.clear();
		}
		
		void checkStep() {	
			for (auto &query : queries)  {
				query->advancePointer(1);
				if (query->isReady()) {
					for (int i = timestampPointer.end -10 ; i <timestampPointer.end;  i++) {
				
				
				}
					int copySize = rdfBuffer.size();
					advanceDevicePointer();				
					query->launch();
				}
			}
		}
		
		void start() {
			
			int counter = 0;

			struct timeval startingTs;
			gettimeofday(&startingTs, NULL);
			long int ts = startingTs.tv_sec * 1000000 + startingTs.tv_usec;

			for (auto &query : queries) {
				query->setStartingTimestamp(ts);
			}
			
			std::cout << "Initial ts is " << ts << std::endl;
			
			for (int i =0; i <srcSize; i++) {
			

				tripleContainer currentTriple  = source[i];
				
				
				struct timeval tp;
				gettimeofday(&tp, NULL);
				long int ms = tp.tv_sec * 1000000 + tp.tv_usec;
				
				timestampPointer.pointer[timestampPointer.end] = ms;
				timestampPointer.end = (timestampPointer.end + 1) % timestampPointer.size;
				timestampPointer.begin = timestampPointer.end;
				
				rdfBuffer.push_back(currentTriple);
				subjectBuffer.push_back(currentTriple.subject);
				predicateBuffer.push_back(currentTriple.predicate);
				objectBuffer.push_back(currentTriple.object);
				
				struct timeval begin;
				gettimeofday(&begin, NULL);
				
				checkStep();

				
				
				usleep(spanTime);
			}
			
			
			//***** REMOVE IN FINAL CODE: ONLY FOR TEST *****
			int copySize = rdfBuffer.size();
			
			if (copySize != 0 ) {
				advanceDevicePointer();
					 
				for (auto &query : queries)  {
					query->launch();
				}
							
	
			}
			

			//***** END REMOVE PART-*****			
		}



};


__global__ void unarySelect (circularBuffer<int> src, int* value, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
		return;
	}	

	int newIndex = (src.begin + index) % src.size;
	
	if (src.pointer[newIndex] == (*value)) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}

__global__ void binarySelect (circularBuffer<int> src1, circularBuffer<int> src2, int* value1, int* value2, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= (abs(src1.end - src1.begin + src1.size) % src1.size) ) {
		return;
	}		

	int newIndex = (src1.begin + index) % src1.size;
	if ((src1.pointer[newIndex] == (*value1)) && (src2.pointer[newIndex] == (*value2))) {
		int add = atomicAdd(size, 1);
		dest[add] = store.pointer[newIndex];
	}
}



/*
* Make multiple select query, with specified comparison condition,
* on a triple store. Both queries and the store are supposed to 
* be already on the device. 
* 
* @param d_selectQueries : the array in which are saved the select values
* @param d_storePointer : pointer on the device to the triple store
* @param storeSize : size of the triple store
* @param comparatorMask : array of triple of comparator that are applied to the queries
*			must be of the size of the d_selectQueries
* @return a vector of type mem_t in which are saved the query results.
*/
std::vector<mem_t<tripleContainer>*> rdfSelect(const std::vector<tripleContainer*> d_selectQueries, 
		deviceCircularBuffer d_pointer,
		const int storeSize, 
		std::vector<int>  arrs) 
{
	standard_context_t context;
	//Initialize elements
	int querySize =  d_selectQueries.size();
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	int* currentSize;
	cudaMalloc(&currentSize, sizeof(int));
	int* zero = (int*) malloc(sizeof(int));
	*zero = 0;

	int* finalResultSize  = (int*) malloc(sizeof(int));

	//Cycling on all the queries
	for (int i = 0; i < querySize; i++) {
		//Save variable to pass to the lambda operator
		tripleContainer* currentPointer = d_selectQueries[i];
		
		mem_t<tripleContainer>* currentResult = new mem_t<tripleContainer>(storeSize, context);

		int gridSize = 300;
	        int blockSize = (storeSize / gridSize) + 1;
		cudaMemcpy(currentSize, zero, sizeof(int), cudaMemcpyHostToDevice);
			
		switch(arrs[i]) {

			case(0): {
				int* value = &(currentPointer->subject);

				unarySelect<<<gridSize,blockSize>>>(d_pointer.subject, value, currentResult->data(), d_pointer.rdfStore, currentSize);

				break;
			}

			case(1): {
				int* value = &(currentPointer->predicate);
			
				unarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, value, currentResult->data(), d_pointer.rdfStore, currentSize);

				break;
			}
						
			case(2): {
                                int* value = &(currentPointer->object);

                                unarySelect<<<gridSize,blockSize>>>(d_pointer.object, value, currentResult->data(), d_pointer.rdfStore, currentSize);
                                break;
			}
			
			case(3): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->predicate);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.predicate, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize);

				break;
			}

			case(4): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize);

				break;
			}

			case(5): {
				int* value1 = &(currentPointer->predicate);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize);

				break;
			}
						
			case(6): {
				cudaMemcpy(currentResult->data(), d_pointer.rdfStore.pointer, storeSize * sizeof(tripleContainer), cudaMemcpyDeviceToDevice);
				cudaMemcpy(currentSize, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
                                break;
			}
			
			
			default: {
				printf("ERROR ERRROR ERROR ERROR ERROR ERROR ERROR");
			}


		}
				
                cudaMemcpy(finalResultSize, currentSize, sizeof(int), cudaMemcpyDeviceToHost);
		currentResult->setSize(*finalResultSize);
		finalResults.push_back(currentResult);
	}
	cudaFree(currentSize);
	
	return finalResults;
}



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
			int tripleA[3] = {a.subject, a.predicate, a.object};
			int tripleB[3] = {b.subject, b.predicate, b.object};
			
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



struct mask_s {
	int subject;
	int predicate;
	int object;
};
 
__global__ void reorderTriple(tripleContainer* src, tripleContainer* dest, int maxSize, mask_s mask) {
		
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}

	int triple[3] = {src[destIndex].subject, src[destIndex].predicate, src[destIndex].object};
 	tripleContainer destTriple = {triple[mask.subject], -1, -1};
 	
 	if (mask.predicate != -1) {
 		destTriple.predicate = triple[mask.predicate];
 	}
 	
 	
 	if (mask.object != -1) {
	 	destTriple.object = triple[mask.object];
 	}
 	
	dest[destIndex] = destTriple;
}


__global__ void indexCopy(tripleContainer* innerSrc, tripleContainer* innerDest, tripleContainer* outerSrc, tripleContainer* outerDest, int2* srcIndex, int maxSize) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
 
	if (destIndex  >= maxSize)  {
		return;
	}
	
	//INVETERD INDEX DUE TO INVERTED JOIN PROBLEM (it should be inner = x, outer = y)
	int innerIndex = srcIndex[destIndex].y;
	int outerIndex = srcIndex[destIndex].x;
	
	innerDest[destIndex] = innerSrc[innerIndex];	
	outerDest[destIndex] = outerSrc[outerIndex];
}



std::vector<mem_t<tripleContainer>*> rdfJoin(tripleContainer* innerTable, int innerSize, tripleContainer* outerTable, int outerSize, JoinMask innerMask[3], JoinMask outerMask[3])
{
	standard_context_t context;
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	TripleSorter* innerSorter = new TripleSorter(innerMask);

	struct timeval beginCu, end;
	gettimeofday(&beginCu, NULL);
	mask_s mask;
	mask.subject = static_cast<int> (outerMask[0]);
	mask.predicate = static_cast<int> (outerMask[1]);
	mask.object = static_cast<int> (outerMask[2]);	
	int gridSize = 64;
	int blockSize = (outerSize/ gridSize) + 1;
	mem_t<tripleContainer>* tempOuter = new mem_t<tripleContainer>(outerSize, context);
	reorderTriple<<<gridSize, blockSize>>>(outerTable, tempOuter->data(), outerSize, mask);
	cudaDeviceSynchronize();
	gettimeofday(&end, NULL);
	float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
	
	std::cout << "OVERHEAD TIME IS " << cuTime << std::endl;
	
	//Sort the two input array
	mergesort(innerTable, innerSize , *innerSorter, context);
	mergesort(tempOuter->data(), outerSize , *innerSorter, context);
	
	
	
	//BUG che mi costringe ad invertire inner con outer?
	mem_t<int2> joinResult = inner_join( innerTable, innerSize, tempOuter->data(), outerSize,  *innerSorter, context);
		
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
        mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	gridSize = 64;
	blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), outerTable, outerResults->data(), joinResult.data(), joinResult.size());

	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);

	return finalResults;
}




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
		deviceCircularBuffer windowPointer;


		std::vector<float> timeCuVector;                
		std::vector<float> timeExVector;


		ifstream rdfStoreFile (argv[1]);
		string strInput;

		int fileLength = 0;	 
		while (std::getline(rdfStoreFile, strInput)) {
			++fileLength;
		}
	
		rdfStoreFile.clear();
		rdfStoreFile.seekg(0, ios::beg);

                size_t rdfSize = fileLength  * sizeof(tripleContainer);
                tripleContainer* h_rdfStore = (tripleContainer*) malloc(rdfSize);


		TripleGenerator manager(h_rdfStore, fileLength, 1, BUFFER_SIZE);

                //read store from rdfStore
                for (int i = 0; i <fileLength; i++) {
			getline(rdfStoreFile,strInput);
                        std::vector<string> triple;
                        separateWords(strInput, triple, ' ');
			
			h_rdfStore[i].subject = atoi(triple[0].c_str());
                        h_rdfStore[i].predicate = atoi(triple[1].c_str());
                        h_rdfStore[i].object = atoi(triple[2].c_str());

                }

                rdfStoreFile.close();

                int N_CYCLE = 1;
		for (int i = 0; i < N_CYCLE; i++) {
			gettimeofday(&beginCu, NULL);
			
			cudaMalloc(&windowPointer.rdfStore.pointer, BUFFER_SIZE * sizeof(tripleContainer));
			cudaMalloc(&windowPointer.subject.pointer, BUFFER_SIZE * sizeof(int));
                        cudaMalloc(&windowPointer.predicate.pointer, BUFFER_SIZE * sizeof(int));
                        cudaMalloc(&windowPointer.object.pointer, BUFFER_SIZE * sizeof(int));
					
			int begin = 0;
			windowPointer.setBegin(begin);
			windowPointer.setEnd(begin);
			windowPointer.setSize(BUFFER_SIZE);
			
			manager.setDevicePointer(windowPointer);
			
		        //set Queries (select that will be joined)
		        tripleContainer h_queryVector1 =  {-1, -1 , 99};
		        tripleContainer h_queryVector2 =	{0, -1, -1}; 

			cout << "query is: first obj: " << h_queryVector1.object << "; secons subj: " << h_queryVector2.subject << endl;

		        mem_t<tripleContainer> d_queryVector1(1, context);
			cudaMemcpy(d_queryVector1.data(), &h_queryVector1, sizeof(tripleContainer), cudaMemcpyHostToDevice);
		
		        mem_t<tripleContainer> d_queryVector2(1, context);
			cudaMemcpy(d_queryVector2.data(), &h_queryVector2, sizeof(tripleContainer), cudaMemcpyHostToDevice);
	
			//set select mask operation
			std::vector<tripleContainer*> selectQuery;
			selectQuery.push_back(d_queryVector1.data());
			selectQuery.push_back(d_queryVector2.data());

			SelectArr arr1 = SelectArr::SPO;
			SelectArr arr2 = SelectArr::SPO;
		
			//set Join mask
			JoinMask innerMask[3];
			innerMask[0] = JoinMask::PRE;
			innerMask[1] = JoinMask::NJ;
			innerMask[2] = JoinMask::NJ;
			
			JoinMask outerMask[3];
			outerMask[0] = JoinMask::PRE;
			outerMask[1] = JoinMask::NJ;
			outerMask[2] = JoinMask::NJ;

			//Creat operation object to pass to query manager
			SelectOperation  selectOp1(&d_queryVector1, arr1);
			SelectOperation  selectOp2(&d_queryVector2, arr2);
		
			JoinOperation  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), innerMask, outerMask);
		
			std::vector<SelectOperation*> selectOperations;
			std::vector<JoinOperation*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			selectOperations.push_back(&selectOp2);
			joinOperations.push_back(&joinOp);
			
			int stepCount = 100000;

			TimeQuery count(selectOperations, joinOperations, windowPointer, manager.getTimestampPointer(), 10000, 10000);
			manager.addQuery(count);
			
		/*	CountQuery count(selectOperations, joinOperations, windowPointer, 150000);
			manager.addQuery(count);*/
			
			gettimeofday(&beginEx, NULL);	
			
			manager.start();

			
			/*//Retrive results from memory
			std::vector<tripleContainer> finalInnerResults = from_mem(*joinOp.getInnerResult());
			std::vector<tripleContainer> finalOuterResults = from_mem(*joinOp.getOuterResult());*/
			
			cudaDeviceSynchronize();
			gettimeofday(&end, NULL);

                      /*  std::vector<tripleContainer> selectResults = from_mem(*selectOp1.getResult());
                        std::vector<tripleContainer> selectResults2 = from_mem(*selectOp2.getResult());*/

			float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
			float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			
			timeCuVector.push_back(cuTime);
			timeExVector.push_back(exTime);
			/*
			//Print Results
			cout << "first select result" << endl;
			for (int i = 0; i < selectResults.size(); i++) {
				cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
			}*/
			

			
			//Print current cycle results
			/*cout <<"first Select Size " << selectResults.size() << endl;
			cout << "second Select Size " << selectResults2.size() << endl;
			cout << "join Size " << finalOuterResults.size() << endl;*/
						
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;					
			cout << "" << endl;


			cudaFree(windowPointer.subject.pointer);
			cudaFree(windowPointer.predicate.pointer);
			cudaFree(windowPointer.object.pointer);
			cudaFree(windowPointer.rdfStore.pointer);
		}

/*	
		std::vector<float> statistics;
		
		statistics = stats<float, float>(timeCuVector);	
                cout << "mean cuda time " << statistics[0] << endl;
                cout << "variance cuda time " << statistics[1] << endl;

                statistics = stats<float, float>(timeExVector);
                cout << "mean ex time " << statistics[0] << endl;
                cout << "variance ex time " << statistics[1] << endl;
*/

		cout << "FINAL VALUE IS " << TEST_VALUE[0] << std::endl;
		cout << "FINAL VALUE IS " << TEST_VALUE[1] << std::endl;
		
                return 0;
}
