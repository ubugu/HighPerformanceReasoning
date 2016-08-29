#pragma once

#include <cstdlib>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "rdfSelect.hxx"
#include "rdfJoin.hxx"



using google::dense_hash_map;


class Query {
	protected:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		CircularBuffer<TripleContainer> windowPointer;

	public:
		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, CircularBuffer<TripleContainer> rdfPointer) {
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
				
				for (int z = 0; z < d_result->table_header.size(); z++) {
					std::cout << "header are " << d_result->table_header[z] << std::endl;
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
	

			for (auto op : join) {}
					
		}
		
	
};


class CountQuery : public Query {
	private:
		int count;
		int currentCount;

	public:
		CountQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			CircularBuffer<TripleContainer> rdfPointer,
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
		CircularBuffer<long int> timestampPointer;
		//TIME IS IN U_SEC
		long int stepTime;
		long int windowTime;
		long int lastTimestamp;
		
	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			CircularBuffer<TripleContainer> rdfPointer, CircularBuffer<long int> timestampPointer,
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
