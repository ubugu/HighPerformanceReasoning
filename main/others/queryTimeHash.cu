#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>
#include <sparsehash/dense_hash_map>

using namespace mgpu;
using google::dense_hash_map;


//TODO implementare la projection su gpu

//TODO 
//VARIABILI PER TESTING, DA RIMUOVERE DAL CODICE FINALE
int VALUE = 0;
std::vector<float> timeCuVector;                
std::vector<long int> timeExVector;
bool isLaunched = false;
dense_hash_map<size_t, std::string> map;

//** END TESTING ***//

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
};

//Struct for containing the pointer to an rdf store (divided into subject predicate and object) 
template<typename rdf_t, typename arr_t>
struct triplePointer {
	rdf_t rdfStore;
	arr_t subject;
	arr_t predicate;
	arr_t object;
};


template <int N>
struct Row 
{
        size_t element[N];
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


/*
* Specific implementation of triplePointer for ciruclar buffer.
* Offers methods for managing the attributes of the class.
*/
struct deviceCircularBuffer : triplePointer<circularBuffer<tripleContainer>, circularBuffer<size_t>> {
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
	//TODO Modificare classe in modo che permetta la join di join
	private:
		Binding** innerTable;
		Binding** outerTable;
		Binding* result = 0;
		
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
		

		Binding* getResult() {
			return this->result;
		};
		
		void setResult(Binding* result) {
			this->result = result;
		};
		

		
		
};

enum class SelectArr { S = 0, P = 1, O = 2, SP = 3, SO = 4, PO = 5, SPO = 6};




__global__ void unarySelect (circularBuffer<tripleContainer> src, int target, int first, int second, size_t* value, size_t* dest, int width, int* size) {

			int index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index >= (abs(src.end - src.begin +  src.size) % src.size) ) {
				return;
			}	

			int newIndex = (src.begin + index) % src.size;
	
			size_t temp[3] = {src.pointer[newIndex].subject, src.pointer[newIndex].predicate, src.pointer[newIndex].object};
	
			if (temp[target] == (*value)) {
				int add = atomicAdd(size, 1);
				size_t* dest_p = (size_t*) (dest + add * width) ;



				*dest_p = temp[first];
				*(dest_p + 1) = temp[second];
		
			}
		}


		__global__ void binarySelect (circularBuffer<size_t> src1, circularBuffer<size_t> src2, size_t* value1, size_t* value2, tripleContainer* dest, circularBuffer<tripleContainer> store, int* size) {

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



class SelectOperationInteface {
	protected:
		mem_t<tripleContainer>* query;
		Binding* result;
		int arr;
		std::vector<std::string> variables;

	public:
		SelectOperationInteface (mem_t<tripleContainer>* query, SelectArr arr, std::string variable) {
			this->query = query;	
			this->arr = static_cast<int> (arr);
			separateWords(variable, variables, ' ');
		};

		int getArr() {
			return this-> arr;
		}
			
		mem_t<tripleContainer>* getQuery() {
			return this->query;
		};
		                                                                            
		
		void setResult(Binding* result) {
			this->result = result;
		};
		
		Binding* getResult() {
			return result;
		};
		
		std::vector<std::string> getVariables() {
			return variables;
		};
		
		Binding** getResultAddress() {
			return &result;
		}


		





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
		void rdfSelect(deviceCircularBuffer d_pointer, const int storeSize) 
		{	
			
		}


};



class SelectOperation 
{
	private:
		mem_t<tripleContainer>* query;
		Binding* result;
		int arr;
		std::vector<std::string> variables;

	public:
		SelectOperation(mem_t<tripleContainer>* query, SelectArr arr, std::string variable) {
			this->query = query;	
			this->arr = static_cast<int> (arr);
			separateWords(variable, variables, ' ');
		};

		int getArr() {
			return this-> arr;
		}
			
		mem_t<tripleContainer>* getQuery() {
			return this->query;
		};
		                                                                            
		
		void setResult(Binding* result) {
			this->result = result;
		};
		
		Binding* getResult() {
			return result;
		};
		
		std::vector<std::string> getVariables() {
			return variables;
		};
		
		Binding** getResultAddress() {
			return &result;
		}


		





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
		void rdfSelect(deviceCircularBuffer d_pointer, const int storeSize) 
		{	
			//Initialize elements	
			int* d_resultSize;
			cudaMalloc(&d_resultSize, sizeof(int));
			int h_resultSize  = 0;

			cudaMemcpy(d_resultSize, &h_resultSize, sizeof(int), cudaMemcpyHostToDevice);
	
			//INSERIRE DIVISIONE CORRETTA
			int gridSize = 300;
			int blockSize = (storeSize / gridSize) + 1;
			tripleContainer* query = this->query->data();
		
			result = new Binding(2, storeSize);
			std::cout << "TARGET ADDRESS IS " << result << std::endl;	
			switch(arr) {

				case(0): {
					size_t* value = &(query->subject);
					unarySelect<<<gridSize,blockSize>>>(d_pointer.rdfStore, 0, 1, 2, value, result->pointer, result->width, d_resultSize);
					break;
				}

				case(1): {
					size_t* value = &(query->predicate);
					unarySelect<<<gridSize,blockSize>>>(d_pointer.rdfStore,  1, 0, 2, value, result->pointer, result->width, d_resultSize);
					break;
				}
					
				case(2): {
				
			                size_t* value = &(query->object);
	
			                unarySelect<<<gridSize,blockSize>>>(d_pointer.rdfStore,  2, 0, 1, value, result->pointer, result->width, d_resultSize);
			             
			                break;
				}
		
		/*		case(3): {
					size_t* value1 = &(query->subject);
					size_t* value2 = &(query->predicate);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.predicate, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}

				case(4): {
					size_t* value1 = &(query->subject);
					size_t* value2 = &(query->object);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.subject, d_pointer.object, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}

				case(5): {
					size_t* value1 = &(query->predicate);
					size_t* value2 = &(query->object);
					binarySelect<<<gridSize,blockSize>>>(d_pointer.predicate, d_pointer.object, value1, value2, result->data(), d_pointer.rdfStore, d_resultSize);
					break;
				}
					
				case(6): {
					cudaMemcpy(result->data(), d_pointer.rdfStore.pointer, storeSize * sizeof(tripleContainer), cudaMemcpyDeviceToDevice);
					cudaMemcpy(d_resultSize, &storeSize, sizeof(int), cudaMemcpyHostToDevice);
			                break;
				}*/
		
			}
			cudaDeviceSynchronize();
	
	
			cudaMemcpy(&h_resultSize, d_resultSize, sizeof(int), cudaMemcpyDeviceToHost);

			result->height  =  h_resultSize;		

			cudaFree(d_resultSize);

		}




	
};

//Sorter for sorting the triple due to theorder defined by the sortMask
template<int N>
 class Sorter {
	private:
		int sortMask[3];
	public:
		Sorter(int sortMask[3]) {
			this->sortMask[0] =  (sortMask[0]);
			this->sortMask[1] =  (sortMask[1]);
			this->sortMask[2] =  (sortMask[2]);
				
		}
		
		
		MGPU_DEVICE bool operator() (Row<N> a, Row<N> b) {
			        	
			if ((sortMask[0] != -1) && (a.element[sortMask[0]] < b.element[sortMask[0]])) {
				return true;
			}
			
			if ((sortMask[1] != -1) && (a.element[sortMask[0]] == b.element[sortMask[0]]) && (a.element[sortMask[1]] < b.element[sortMask[1]])) {
				return true;
			}
			
			if ((sortMask[2] != -1) && (a.element[sortMask[0]] == b.element[sortMask[0]]) && (a.element[sortMask[1]] == b.element[sortMask[1]]) && (a.element[sortMask[2]] < b.element[sortMask[2]])) {
				return true;
			}
			
			return false;
		}
};


 
template<int N>
__global__ void typeCopy(Row<N>* dest, size_t* src, int* pos, int maxSize, int width) {
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (destIndex  >= maxSize)  {
		return;
	}

	dest[threadIdx.x].element[pos[blockIdx.x]] = src[threadIdx.x * width + blockIdx.x];
}



__global__ void indexCopy(/*tripleContainer* innerSrc, tripleContainer* innerDest, tripleContainer* outerSrc, tripleContainer* outerDest, int2* srcIndex, int maxSize*/) 
{
	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	printf("BOH \n");
 /*
	if (destIndex  >= maxSize)  {
		return;
	}
	/*
	int innerIndex = srcIndex[destIndex].x;
	int outerIndex = srcIndex[destIndex].y;
	
	innerDest[destIndex] = innerSrc[innerIndex];	
	outerDest[destIndex] = outerSrc[outerIndex];*/
}

template<int N>
__global__ void printTest(Row<N>* src1, Row<N>* src2, int2* indexes) {

	int destIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	printf("INDeXES ARE %i %i VALUES ARE %i  %i %i %i \n", indexes[destIndex].x, indexes[destIndex].y, src1[indexes[destIndex].x].element[0], src1[indexes[destIndex].x].element[1], src2[indexes[destIndex].y].element[0], src2[indexes[destIndex].y].element[1]);
}



template<int N>
std::vector<mem_t<tripleContainer>*> rdfJoin(Binding* innerTable, Binding* outerTable, std::vector<std::string> joinMask)
{
	//TODO Migliorare la join nel reordering delle triple
	standard_context_t context;
	std::vector<mem_t<tripleContainer>*> finalResults;
	
	Row<N>* tempInner;
	cudaMalloc(&tempInner, innerTable->height * sizeof(Row<N>));
	
	Row<N>* tempOuter;
	cudaMalloc(&tempOuter, outerTable->height * sizeof(Row<N>));
	
	int* d_innerPos;
	cudaMalloc(&d_innerPos, innerTable->width * sizeof(int));
	int* d_outerPos;
	cudaMalloc(&d_outerPos, outerTable->width * sizeof(int));
	
	int* h_innerPos = (int*) malloc(innerTable->width * sizeof(int));
	h_innerPos[0] = 0;
	h_innerPos[1] = 1;
	int* h_outerPos = (int*) malloc(outerTable->width * sizeof(int));
	h_outerPos[0] = 1;
	h_outerPos[1] = 0;

	cudaMemcpy(d_innerPos, h_innerPos, innerTable->width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_outerPos, h_outerPos, outerTable->width * sizeof(int), cudaMemcpyHostToDevice);
	
	
	typeCopy<<<innerTable->width, innerTable->height>>>(tempInner, innerTable->pointer, d_innerPos, innerTable->height * innerTable->width, innerTable->width);
	typeCopy<<<outerTable->width, outerTable->height>>>(tempOuter, outerTable->pointer, d_outerPos, outerTable->height * outerTable->width, outerTable->width);
	
		
	std::cout << "LAUNCH MEGER " << std::endl;	
	int mask[3] = {0, -1, -1};
	
	Sorter<N>* sorter = new Sorter<N>(mask);
		

	
	std::cout << "LAUNCH MEGER " << std::endl;

	//Sort the two input array
	mergesort<launch_params_t<128, 2>>(tempInner, innerTable->height, *sorter, context);
	mergesort<launch_params_t<128, 2>>(tempOuter, outerTable->height , *sorter, context);

	std::cout << "OUTER HEIGH IS " << innerTable->height << std::endl;
	Binding* d_result = innerTable;
	size_t* final_binding = (size_t*) malloc(d_result->height * d_result->width * sizeof(size_t));
	cudaMemcpy(final_binding, d_result->pointer, d_result->width * sizeof(size_t) * d_result->height, cudaMemcpyDeviceToHost);
	
	for (int i =0; i <10; i+=2) {
		std::cout << "original is " << map[ final_binding[i]] << " " <<  map[final_binding[i + 1]] << std::endl;
	}	
	
	Row<N>* testR  = (Row<N>*) malloc(innerTable->height * sizeof(Row<N>)); 
	cudaMemcpy(testR, tempInner, innerTable->height * sizeof(Row<N>), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 5; i++) {
//		std::cout << "result is " << map[testR[i].element[0]] << " second " <<  map[testR[i].element[1]] << std::endl;
		std::cout << "result is " << testR[i].element[0] << " second " <<  testR[i].element[1] << std::endl;
	}
	


	mem_t<int2> joinResult = inner_join<launch_params_t<128,2>>( tempInner, innerTable->height, tempOuter, outerTable->height,  *sorter, context);
		
	std::cout << "JOIN RESULT SIZE IS " << joinResult.size() << std::endl;
	printTest<<<1, joinResult.size()>>>(tempInner, tempOuter, joinResult.data());

	
	cudaDeviceSynchronize();
	std::cout << "LAUNCHED"<< std::endl;
	exit(1);
	/*
	mem_t<tripleContainer>* innerResults = new mem_t<tripleContainer>(joinResult.size(), context);
    mem_t<tripleContainer>* outerResults = new mem_t<tripleContainer>(joinResult.size(), context);
	
	//SETTARE DIVISIONE CORRETTA
	//TODO BIsogna settare come comporatrsi quando il valore della join supera i 129k risultati
	gridSize = 64;
	blockSize = (joinResult.size() / gridSize) + 1; 
	indexCopy<<<gridSize, blockSize>>>(innerTable, innerResults->data(), tempOuter->data(), outerResults->data(), joinResult.data(), joinResult.size());
	
	cudaFree(tempOuter);
	cudaFree(tempInner);
	
	*/

	return finalResults;
}


class Query {
	protected:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		deviceCircularBuffer windowPointer;

	public:
		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, deviceCircularBuffer rdfPointer) {
			this->join = join;
			this->select = select;
			this->windowPointer = rdfPointer;
		}

		virtual void setWindowEnd(int step) {
			windowPointer.setEnd(step);
		}
		
		/**
		* Function for managing query execution
		**/
		//TODO Verificare se si puo migliorare
		void startQuery() {
			int storeSize =  windowPointer.rdfStore.getLength();			
			
			for (auto op : select) {
					op->rdfSelect(windowPointer, storeSize);

				}
			
	
			for (int i = 0; i < join.size(); i++) {
				Binding* innerTable = join[i]->getInnerTable();
				Binding* outerTable = join[i]->getOuterTable();
			
				int width = 2;
			
				switch(width) {
					case(1): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<1>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(2): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<2>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}
/*
					case(3): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<3>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(4): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<4>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(5): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<5>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(6): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<6>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(7): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<7>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(8): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<8>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(9): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<9>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}

					case(10): {
						std::vector<mem_t<tripleContainer>*>  joinResult = rdfJoin<10>(innerTable, outerTable, join[i]->getJoinMask());
						break;
					}*/
				}

			/*	join[i]->setInnerResult(joinResult[0]);
				join[i]->setOuterResult(joinResult[1]);	*/		
			}
			
			exit(1);
	
		}

		//TODO modificare quando si sapra come utilizzare i risultati
		void printResults(dense_hash_map<size_t, std::string> map) {

			int w = 0;
			for (auto op : select) {
				printf("W VALUE IS %i \n", w);
				if (w == 0) VALUE += op->getResult()->height;
				
				Binding* d_result = op->getResult();
				
				size_t* final_binding = (size_t*) malloc(d_result->height * d_result->width * sizeof(size_t));
				cudaMemcpy(final_binding, d_result->pointer, d_result->width * sizeof(size_t) * d_result->height, cudaMemcpyDeviceToHost);
				std::cout << "size is " << d_result->height << std::endl;
				std::cout << "width is " << d_result->width << std::endl;
				
				/*for (int z = 0; z < d_result->header.size(); z++) {
				std::cout << "header are " << d_result->header[z] << std::endl;
				}*/
				
				for (int i =0; i < d_result->height; i++) {
					//for (int k = 0; k < d_result->width; k++) {
						//size_t current = final_binding[i + k];
						std::cout << "result is " << map[ final_binding[i]] << " " <<  map[final_binding[i + 1]] << std::endl;
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
					innerHash.push_back( map[innerRes[i].subject]);
                                        innerHash.push_back( map[innerRes[i].predicate]);
                                        innerHash.push_back( map[innerRes[i].object]);

				}
				
				std::vector<tripleContainer> outerRes = from_mem(*op->getOuterResult());
				std::vector<const char*> outerHash;
				for (int i =0; i< outerRes.size(); i++) {
					outerHash.push_back( map[outerRes[i].subject]);
                                        outerHash.push_back( map[outerRes[i].predicate]);
                                        outerHash.push_back( map[outerRes[i].object]);

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
			deviceCircularBuffer rdfPointer,
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
		long int stepTime;
		long int windowTime;
		long int lastTimestamp;
		
	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			deviceCircularBuffer rdfPointer, circularBuffer<long int> timestampPointer,
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
			windowPointer.setBegin(newBegin);
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



class QueryManager {
	private:
		int spanTime;
		std::string* source;
		int srcSize;
		
		std::vector<tripleContainer> rdfBuffer;
		std::vector<size_t> subjectBuffer;
		std::vector<size_t> predicateBuffer;
		std::vector<size_t> objectBuffer;

		std::vector<TimeQuery> timeQueries;
		std::vector<CountQuery> countQueries;
		
		circularBuffer<long int> timestampPointer;
		deviceCircularBuffer devicePointer;
    //            dense_hash_map<size_t, std::string> map;


	public:
		QueryManager(std::string* source, int srcSize, int spanTime,int buffSize)   {
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
				
		void addTimeQuery(TimeQuery query) {
			timeQueries.push_back(query);
		}
		
		void addCountQuery(CountQuery query) {
			countQueries.push_back(query);
		}

				
		void copyElements (int deviceSpan, int hostSpan, int copySize) {
			cudaMemcpy(deviceSpan + devicePointer.rdfStore.pointer, &rdfBuffer[0] + hostSpan, copySize * sizeof(tripleContainer), cudaMemcpyHostToDevice); 
			cudaMemcpy(deviceSpan + devicePointer.subject.pointer, &subjectBuffer[0] + hostSpan, copySize * sizeof(size_t), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceSpan + devicePointer.predicate.pointer,&predicateBuffer[0] + hostSpan, copySize * sizeof(size_t), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceSpan + devicePointer.object.pointer, &objectBuffer[0] + hostSpan, copySize * sizeof(size_t), cudaMemcpyHostToDevice);
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
			for (auto &query : countQueries)  {
				query.incrementCount();
				if (query.isReady()) {
					advanceDevicePointer();
					query.setWindowEnd(devicePointer.rdfStore.end);			
					query.launch();
					query.printResults(map);
				}
			}
			
			for (auto &query : timeQueries) {
				if (query.isReady(timestampPointer.pointer[timestampPointer.end - 1])) {
					advanceDevicePointer();
					query.setWindowEnd(devicePointer.rdfStore.end - 1);		
					query.launch();
					query.printResults(map);
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

			map.set_empty_key(NULL);       
		

			for (int i =0; i <srcSize; i++) {

				
				tripleContainer currentTriple;
 
                                std::vector<std::string> triple;
                                separateWords(source[i], triple, ' ');
			
			        currentTriple.subject = h_func(triple[0]);
                                currentTriple.predicate = h_func(triple[1]);
                                currentTriple.object = h_func(triple[2]);

				map[currentTriple.subject] = triple[0];
                                map[currentTriple.predicate] = triple[1];
                                map[currentTriple.object] = triple[2] ;

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
							
				checkStep();
			//	usleep(spanTime);

			}
			
			//TODO vedere se occorre tenere o no quest'ultima parte
			//***** REMOVE IN FINAL CODE: ONLY FOR TEST (FORSE) *****
			//DA OTTIMIZZARE POICHE VIENE LANCIATO ANCHE QUANDO NON SERVE!!
			advanceDevicePointer();	 
			for (auto &query : timeQueries)  {
				query.setWindowEnd(devicePointer.rdfStore.end);
				query.launch();
				query.printResults(map);
			}
			
			for (auto &query :countQueries) {
				query.setWindowEnd(devicePointer.rdfStore.end);
				query.launch();
				query.printResults(map);
			}
			//***** END REMOVE PART-*****			
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
		deviceCircularBuffer windowPointer;




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

                int N_CYCLE = 1;
		for (int i = 0; i < N_CYCLE; i++) {
			
			gettimeofday(&beginCu, NULL);
			QueryManager manager(h_rdfStore, fileLength, 1, BUFFER_SIZE);
			cudaMalloc(&windowPointer.rdfStore.pointer, BUFFER_SIZE * sizeof(tripleContainer));
			cudaMalloc(&windowPointer.subject.pointer, BUFFER_SIZE * sizeof(size_t));
                        cudaMalloc(&windowPointer.predicate.pointer, BUFFER_SIZE * sizeof(size_t));
                        cudaMalloc(&windowPointer.object.pointer, BUFFER_SIZE * sizeof(size_t));
					
			int begin = 0;
			windowPointer.setBegin(begin);
			windowPointer.setEnd(begin);
			windowPointer.setSize(BUFFER_SIZE);
			
			manager.setDevicePointer(windowPointer);
			basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;
		        //set Queries (select that will be joined)
		        tripleContainer h_queryVector1;
			h_queryVector1.subject = 0;
			h_queryVector1.predicate = 0;
			h_queryVector1.object =  h_func("<http://example.org/int/99>");
			
			
		        tripleContainer h_queryVector2;	
			h_queryVector2.subject = h_func("<http://example.org/int/0>");
			h_queryVector2.predicate = 0;
			h_queryVector2.object = 0;

			cout << "query is: first obj: " << h_queryVector1.object << "; secons subj: " << h_queryVector2.subject << endl;

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
		

			//Creat operation object to pass to query manager
			SelectOperation  selectOp1(&d_queryVector1, arr1, "?s ?p");
			SelectOperation  selectOp2(&d_queryVector2, arr2, "?p ?o");
		
			JoinOperation  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), "?p");
		
			std::vector<SelectOperation*> selectOperations;
			std::vector<JoinOperation*> joinOperations;
		
			selectOperations.push_back(&selectOp1);
			selectOperations.push_back(&selectOp2);
			joinOperations.push_back(&joinOp);
			
			int stepCount = 100000;
			//std::cout << "starting tmsp " << manager.getTimestampPointer().begin << std::endl;
			
			
			
			/*TimeQuery count(selectOperations, joinOperations, windowPointer, manager.getTimestampPointer(), 5000, 5000);
			manager.addTimeQuery(count);


			TimeQuery count5(selectOperations, joinOperations, windowPointer, manager.getTimestampPointer(), 7000, 7000);
			manager.addTimeQuery(count5);*/
			
			CountQuery count2(selectOperations, joinOperations, windowPointer, 50000);
			manager.addCountQuery(count2);
			
			
			/*CountQuery count3(selectOperations, joinOperations, windowPointer, 50000);
			manager.addCountQuery(count3);
			
			CountQuery count4(selectOperations, joinOperations, windowPointer, 34652);
			manager.addCountQuery(count4);	*/		
			
			gettimeofday(&beginEx, NULL);	

			manager.start();


			
			cudaDeviceSynchronize();
			gettimeofday(&end, NULL);


			float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
			float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
			float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
			

			
			timeCuVector.push_back(cuTime);

						
			cout << "Total time: " << prTime << endl;
			cout << "Cuda time: " << cuTime << endl;
			cout << "Execution time: " << exTime << endl;					
			cout << "" << endl;


			cudaFree(windowPointer.subject.pointer);
			cudaFree(windowPointer.predicate.pointer);
			cudaFree(windowPointer.object.pointer);
			cudaFree(windowPointer.rdfStore.pointer);
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

