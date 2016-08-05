#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>

using namespace mgpu;

//struct to contains a single triple with int type.
struct tripleContainer {
        int subject;
        int predicate;
        int object;
};

template<typename type_t>
struct circularBuffer {
	type_t* pointer;
	int begin;
	int end;
	int size;
};

template<typename rdf_t, typename arr_t>
struct triplePointer {
	rdf_t rdfStore;
	arr_t subject;
	arr_t predicate;
	arr_t object;
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


class Query {
	public:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> windowPointer;

	
		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, 
			triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> rdfPointer) {

			this->join = join;
			this->select = select;
			this->windowPointer = rdfPointer;
		}

		std::vector<SelectOperation*> getSelect() {
			return select;
		}

		std::vector<JoinOperation*> getJoin() {
			return join;
		}

		void advancePointer(int step) {
			int newEnd = (windowPointer.rdfStore.end + step) % windowPointer.rdfStore.size;
			
			windowPointer.rdfStore.end = newEnd;
			windowPointer.subject.end = newEnd;
			windowPointer.predicate.end = newEnd;
			windowPointer.object.end = newEnd;
		}

		triplePointer<tripleContainer*, int*> getStorePointer() {
			triplePointer<tripleContainer*, int*> pointer;
			pointer.rdfStore = windowPointer.rdfStore.pointer;
			pointer.subject = windowPointer.subject.pointer;
			pointer.predicate = windowPointer.predicate.pointer;
			pointer.object = windowPointer.object.pointer;
			
			return pointer;
		}
	
};

void queryManager(std::vector<SelectOperation*> selectOp, std::vector<JoinOperation*> joinOp, 
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> d_pointer, 
		const int storeSize);

class CountQuery : public Query {
	private:
		int count;
		int currentCount;

	public:
		CountQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> rdfPointer,
			int count) : Query(select, join, rdfPointer) {

				this->count = count;
				this->currentCount = 0;
		} 

		void incrementCount() {
			this->currentCount++;
		}

		bool isReady() {
			if (currentCount == count) {
				count = 0;
				return true;
			}
			return false;
		}
	
		void launch() {
			queryManager(getSelect(), getJoin(), windowPointer, count);
			int newBegin = (windowPointer.rdfStore.begin + count) % windowPointer.rdfStore.size;
			windowPointer.rdfStore.begin = newBegin;
			windowPointer.subject.begin = newBegin;
			windowPointer.predicate.begin = newBegin;
			windowPointer.object.begin = newBegin;
 
		}
};

class TimeQuery : public Query {
	private:
		circularBuffer<long int> timestampPointer;
		int stepTime;
		int windowTime;
		long int lastTimestamp;
		bool isFirst;

	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> rdfPointer,
			circularBuffer<long int> timestampPointer,
			int windowTime, int stepTime) : Query(select, join, rdfPointer) {
				this->stepTime = stepTime;
				this->windowTime = windowTime;
				this->lastTimestamp = 0;
				this->isFirst = 0;
				this->timestampPointer = timestampPointer;
		}
		
		void setStartingTimestamp(long int timestamp) {
			this->lastTimestamp = timestamp;
		}

		void advancePointer(int step) : advancePointer(int) {
			timestampPointer.end = (windowPointer.timestampPointer.end + size) % windowPointer.timestampPointer.size;
		}

		

		bool isReady(long int newTimestamp) {
			if (isFirst) {
				if (lastTimestmp + windowTime <= newTimestamp) {
					lastTimestamp += spanTime;
					return true;
				}

			} else {
				if (lastTimestamp + stepTime <= newTimestamp {
					lastTimestamp += spanTime;
					return true;
			}
			
			return false;
		}

		long int* getTimestampPointer() {
			return this->timestampPointer.pointer;
		}

		void launch() {
			
			if (!isFirst) {
				long int startingTime = lastTimestamp - (long int) windowTime;					
				int newBegin = 0;

				if (timestampPointer.end < timestampPointer.begin) {
					bool found = false;

					for(int i = timestampPointer.begin; i < timestampPointer.size; i++) {
						if (timestampPointer.pointer[i] >= startingTime) {
							found = true;
							newBegin = i;
							break;
						}
					}				

					if (found == false) {
						for(int i = 0; i < timestampPointer.end; i++) {
							if (timestampPointer.pointer[i] >= startingTime) {
								newBegin = i;
								break;
							}				
						}				
					}

				} else {
					for (int i = 0; i < timestampPointer.end; i++) {
						if (timestampPointer.pointer[i] >= startingTime) {
							newBegin = i;
							break;
						}
					}

				}

				windowPointer.rdfStore.begin = newBegin;
				windowPointer.subject.begin = newBegin;
				windowPointer.predicate.begin = newBegin;
				windowPointer.object.begin = newBegin;

			} else {
				isFirst = false;
			}	

			

			int diff = windowPointer.rdfStore.end - windowPointer.rdfStore.begin;
			int currentSize = (diff >= 0 ? diff : windowPointer.rdfStore.size + diff); 
			queryManager(getSelect(), getJoin(), windowPointer, currentSize);  
		}


};


class TripleGenerator {
	private:
		int spanTime;
		std::ifstream rdfStoreFile;

		std::vector<tripleContainer> rdfBuffer;
		std::vector<int> subjectBuffer;
		std::vector<int> predicateBuffer;
		std::vector<int> objectBuffer;

		std::vector<CountQuery> countQueries;
		std::vector<TimeQuery> timeQueries;

		circularBuffer<long int> timestampPointer;
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> devicePointer;

	public:
		void checkStep();
		void advanceDevicePointer();
		
		TripleGenerator(int spanTime, std::ifstream inputFile)   {
			this->spanTime = spanTime;
			this->rdfStoreFile = inputFile; 
			long int* timestamp = (long int*) malloc(buffSize * sizeof(long int));
		}
		
		void start() {
			std::string strInput;
 
			while (std::getline(rdfStoreFile, strInput)) {
				std::vector<std::string> triple;
		                separateWords(strInput, triple, ' ');

				tripleContainer currentTriple  = {atoi(triple[0].c_str()), atoi(triple[1].c_str()), atoi(triple[2].c_str())};
				
				struct timeval tp;
				gettimeofday(&tp, NULL);
				long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

				timestapPointer.pointer[timestampPointer.end] = ms;
				timestampPointer.end = (windowPointer.timestampPointer.end + 1) % windowPointer.timestampPointer.size;
				timestampPointer.begin = timestampPointer.end;

				rdfBuffer.push_back(currentTriple);
				subjectBuffer.push_back(currentTriple.subject);
				predicateBuffer.push_back(currentTriple.predicate);
				objectBuffer.push_back(currentTriple.object);

				checkStep();
				
				usleep(spanTime);
			}			
		}

		void advanceDevicePointer() {
			int copySize = rdfBuffer.size();
			
			circularBuffer<tripleContainer> rdfBuff = devicePointer.rdfStore;

			int newEnd = (rdfBuff.end + copySize) % rdfBuff.size;

			if (newEnd < rdfBuff.end) {
				int finalHalf = rdfBuff.size - rdfBuff.end;
				cudaMemcpy(devicePointer.rdfStore.end + devicePointer.rdfStore.pointer, rdfBuffer, finalHalf, cudaMemcpyHostToDevice); 
				cudaMemcpy(devicePointer.subject.end + devicePointer.subject.pointer, subjectBuffer, finalHalf, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.predicate.end + devicePointer.predicate.pointer, predicateBuffer, finalHalf, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.object,end + devicePointer.object.pointer, objectBuffer, finalHalf, cudaMemcpyHostToDevice);			
	
				int firstHalf = copySize - finalHalf;
				cudaMemcpy(devicePointer.rdfStore.pointer, rdfBuffer, firstHalf, cudaMemcpyHostToDevice); 
				cudaMemcpy(devicePointer.subject.pointer, subjectBuffer, firstHalf, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.predicate.pointer, predicateBuffer, firstHalf, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.object.pointer, objectBuffer, firstHalf, cudaMemcpyHostToDevice);			
			} else {
				cudaMemcpy(devicePointer.rdfStore.end + devicePointer.rdfStore.pointer, rdfBuffer, copySize, cudaMemcpyHostToDevice); 
				cudaMemcpy(devicePointer.subject.end + devicePointer.subject.pointer, subjectBuffer, copySize, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.predicate.end + devicePointer.predicate.pointer, predicateBuffer, copySize, cudaMemcpyHostToDevice);
				cudaMemcpy(devicePointer.object,end + devicePointer.object.pointer, objectBuffer, copySize, cudaMemcpyHostToDevice);			
			}


			devicePointer.rdfStore.end = newEnd;
			devicePointer.subject.end = newEnd;
			devicePointer.predicate.end = newEnd;
			devicePointer.object.end = newEnd;

			rdfBuffer.clear();
			subjectBuffer.clear();
			predicateBuffer.clear();
			objectBuffer.clear();
		}
		
		void checkStep() {
			
			for (CountQuery query : countQueries)  {
				query.incrementCount();
				if (query.isReady()) {
					advanceDevicePointer();
					int copySize = rdfBuffer.size();
			
					for (CountQuery other : countQueries)  {
						other.advancePointer(copySize);
					}
				
					for (TimeQuery other : timeQueries) {
						other.advancePointer(copySize);
					}

					query.launch();
				}

			}

			for (TimeQuery query : timeQueries) {
				if (query.isRedy(timestamps.pointer[timestamps.end - 1])) {
					advanceDevicePointer();
					int copySize = rdfBuffer.size();
			
					for (CountQuery query : countQueries)  {
						query.advancePointer(copySize);
					}
				
					for (TimeQuery query : timeQueries) {
						query.advancePointer(copySize);
					}

					query.launch();

				}
			}
		}



};


__global__ void unarySelect (circularBuffer<int> src, int* value, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size, int storeSize) {
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

__global__ void binarySelect (circularBuffer<int> src1, circularBuffer<int> src2, int* value1, int* value2, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size, int storeSize) {
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
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> d_pointer,
		const int storeSize, 
		std::vector<int*> comparatorMask,
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

				unarySelect<<<gridSize,blockSize>>>(d_pointer.subject, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(1): {
				int* value = &(currentPointer->predicate);
			
				unarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}
						
			case(2): {
                                int* value = &(currentPointer->object);

                                unarySelect<<<gridSize,blockSize>>>(d_pointer.object, value, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);
                                break;
			}
			
			case(3): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->predicate);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.predicate, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(4): {
				int* value1 = &(currentPointer->subject);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

				break;
			}

			case(5): {
				int* value1 = &(currentPointer->predicate);
				int* value2 = &(currentPointer->object);

				binarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, d_pointer.object, value1, value2, currentResult->data(), d_pointer.rdfStore, currentSize, storeSize);

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


//Computes inner < outer when doing join to find the elements that needs to be joined
class TripleComparator
{
	private:
		int maskA[3];
		int maskB[3];
	public:
		TripleComparator(JoinMask innerMask[3], JoinMask outerMask[3]) {
			maskA[0] = static_cast<int> (innerMask[0]);
			maskA[1] = static_cast<int> (innerMask[1]);
			maskA[2] = static_cast<int> (innerMask[2]);
			
			maskB[0] = static_cast<int> (outerMask[0]);
			maskB[1] = static_cast<int> (outerMask[1]);
			maskB[2] = static_cast<int> (outerMask[2]);			
		}
		
		MGPU_DEVICE bool operator() (tripleContainer a, tripleContainer b) {			
			int tripleA[3] = {a.subject, a.predicate, a.object};
			int tripleB[3] = {b.subject, b.predicate, b.object};

			if ((maskA[0] != -1) && (tripleA[maskA[0]] < tripleB[maskB[0]])) {
				return true;
			}
		
			if ((maskA[1] != -1) && (tripleA[maskA[0]] == tripleB[maskB[0]]) && (tripleA[maskA[1]] < tripleB[maskA[1]])) {
				return true;
			}
			
			if ((maskA[2] != -1) && (tripleA[maskA[0]] == tripleB[maskB[0]]) && (tripleA[maskA[1]] == tripleB[maskA[1]]) && (tripleA[maskA[2]] < tripleB[maskA[2]])) {
				return true;
			}
		
			return false;
		}
};


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
	TripleSorter* outerSorter = new TripleSorter(outerMask);
	
	//Sort the two input array
	mergesort(innerTable, innerSize , *innerSorter, context);
	mergesort(outerTable, outerSize , *outerSorter, context);
	
	TripleComparator* comparator = new TripleComparator(innerMask, outerMask);
	
	//BUG che mi costringe ad invertire inner con outer?
	mem_t<int2> joinResult = inner_join(outerTable, outerSize, innerTable, innerSize,  *comparator, context);
		
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
        mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	int gridSize = 64;
	int blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), outerTable, outerResults->data(), joinResult.data(), joinResult.size());

	finalResults.push_back(innerResults);
	finalResults.push_back(outerResults);

	return finalResults;
}



/**
* Function for managing query execution
**/
void queryManager(std::vector<SelectOperation*> selectOp, std::vector<JoinOperation*> joinOp, 
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> d_pointer, 
		const int storeSize) {

	std::vector<tripleContainer*> d_selectQueries;
	std::vector<int*> comparatorMask;
	std::vector<int> arrs;

	for (int i = 0; i < selectOp.size(); i++) {
		d_selectQueries.push_back(selectOp[i]->getQuery()->data());
		arrs.push_back(selectOp[i]->getArr());
	}
	

	std::vector<mem_t<tripleContainer>*> selectResults = rdfSelect(d_selectQueries, d_pointer, storeSize, comparatorMask, arrs);

	for (int i = 0; i < selectResults.size(); i++) {
		selectOp[i]->setResult(selectResults[i]);
	}
	
	
	for (int i = 0; i < joinOp.size(); i++) {
		mem_t<tripleContainer>* innerTable = *joinOp[i]->getInnerTable();
		mem_t<tripleContainer>* outerTable = *joinOp[i]->getOuterTable();
		std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin(innerTable->data(), innerTable->size(), outerTable->data(), outerTable->size(), joinOp[i]->getInnerMask(), joinOp[i]->getOuterMask());
		joinOp[i]->setInnerResult(joinResult[0]);
		joinOp[i]->setOuterResult(joinResult[1]);				
	}
	
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
		triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> windowPointer;

		std::vector<float> timeCuVector;                
		std::vector<float> timeExVector;
		std::vector<int> firstVector;
		std::vector<int> secondVector;
		std::vector<int> resultVector;
                int N_CYCLE = 1;
		for (int i = 0; i < N_CYCLE; i++) {
			gettimeofday(&beginCu, NULL);

			cudaMalloc(windowPointer.rdfStore.pointer, BUFFER_SIZE * sizeof(tripleContainer));
			cudaMalloc(windowPointer.subject.pointer, BUFFER_SIZE * sizeof(int));
                        cudaMalloc(windowPointer.predicate.pointer, BUFFER_SIZE * sizeof(int));
                        cudaMalloc(windowPointer.object.pointer, BUFFER_SIZE * sizeof(int));
		
		        //set Queries (select that will be joined)
		        tripleContainer h_queryVector1 =  {-1, -1 , 99};
		        tripleContainer h_queryVector2 =	{0, -1, -1}; 

			cout << "query is " << h_queryVector1.object << " " << h_queryVector2.subject << endl;

		        mem_t<tripleContainer> d_queryVector1(1, context);
			cudaMemcpy(d_queryVector1.data(), &h_queryVector1, sizeof(tripleContainer), cudaMemcpyHostToDevice);
		
		        mem_t<tripleContainer> d_queryVector2(1, context);
			cudaMemcpy(d_queryVector2.data(), &h_queryVector2, sizeof(tripleContainer), cudaMemcpyHostToDevice);
	
			//set select mask operation
			std::vector<tripleContainer*> selectQuery;
			selectQuery.push_back(d_queryVector1.data());
			selectQuery.push_back(d_queryVector2.data());

			SelectArr arr1 = SelectArr::O;

			SelectArr arr2 = SelectArr::S;
			
		
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
		
			gettimeofday(&beginEx, NULL);	
			
			queryManager(selectOperations, joinOperations, buffer->getPointer(), fileLength);
			
			//Retrive results from memory
			std::vector<tripleContainer> finalInnerResults = from_mem(*joinOp.getInnerResult());
			std::vector<tripleContainer> finalOuterResults = from_mem(*joinOp.getOuterResult());
			
			cudaDeviceSynchronize();
			gettimeofday(&end, NULL);

                        std::vector<tripleContainer> selectResults = from_mem(*selectOp1.getResult());
                        std::vector<tripleContainer> selectResults2 = from_mem(*selectOp2.getResult());

			float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
			float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			
			timeCuVector.push_back(cuTime);
			timeExVector.push_back(exTime);
			firstVector.push_back(selectResults.size());
			secondVector.push_back(selectResults2.size());
			resultVector.push_back(finalOuterResults.size());

			/*
			//Print Results
			cout << "first select result" << endl;
			for (int i = 0; i < selectResults.size(); i++) {
				cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
			}
		
			cout << "second select result" << endl;
			for (int i = 0; i < selectResults2.size(); i++) {
				cout << selectResults2[i].subject << " " << selectResults2[i].predicate << " "  << selectResults2[i].object << endl; 
			}
		
			cout << "final inner result" << endl;
			for (int i = 0; i < finalInnerResults.size(); i++) {
			cout << finalInnerResults[i].subject << " " << finalInnerResults[i].predicate << " "  << finalInnerResults[i].object << endl; 
			} 
			
			cout << "final inner result" << endl;
			for (int i = 0; i < finalOuterResults.size(); i++) {
				cout << finalOuterResults[i].subject << " " << finalOuterResults[i].predicate << " "  << finalOuterResults[i].object << endl; 
			} */
			
			//Print current cycle results
			cout <<"first Select Size " << selectResults.size() << endl;
			cout << "second Select Size " << selectResults2.size() << endl;
			cout << "join Size " << finalOuterResults.size() << endl;
						
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;					
			cout << "" << endl;

			cudaFree((*joinOp.getInnerResult()).data());
			cudaFree((*joinOp.getOuterResult()).data());
			cudaFree((*selectOp1.getResult()).data());
			cudaFree((*selectOp2.getResult()).data());
			cudaFree(d_storeVector);
			cudaFree(d_storeVector);
			cudaFree(d_subject);
			cudaFree(d_object);
			cudaFree(d_predicate);
		}
		
		std::vector<float> statistics;
		
		statistics = stats<float, float>(timeCuVector);	
                cout << "mean cuda time " << statistics[0] << endl;
                cout << "variance cuda time " << statistics[1] << endl;

                statistics = stats<float, float>(timeExVector);
                cout << "mean ex time " << statistics[0] << endl;
                cout << "variance ex time " << statistics[1] << endl;
/*
                statistics = stats<int, longlong>(firstVector);
                cout << "mean first select size " << statistics[0] << endl;
                cout << "variance first select size " << statistics[1] << endl;

                statistics = stats<int, longlong>(secondVector);
                cout << "mean second select size " << statistics[0] << endl;
                cout << "variance second select size " << statistics[1] << endl;
	 
                statistics = stats<int, longlong>(resultVector);
                cout << "mean join size " << statistics[0] << endl;
                cout << "variance join size " << statistics[1] << endl;
*/
                return 0;

}



