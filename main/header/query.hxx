#pragma once

#include <cstdlib>
#include <fstream>
#include <unordered_set>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "rdfSelect.hxx"
#include "rdfJoin.hxx"


class Query {
	protected:
		std::vector<SelectOperation*> select;
		std::vector<JoinOperation*> join;
		CircularBuffer<TripleContainer> windowPointer;
		std::vector<std::string> variables_projection;

	public:
		Query(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join, CircularBuffer<TripleContainer> rdfPointer, std::vector<std::string> variables) {
			this->join = join;
			this->select = select;
			this->windowPointer = rdfPointer;
			this->variables_projection = variables;
		}

		virtual void setWindowEnd(int step) {
			windowPointer.end = step;
		}
		
		/**
		* Function for managing query execution
		**/
		void startQuery() {
			int storeSize =  windowPointer.getLength();			

			for (auto op : select) {
				op->rdfSelect(windowPointer, storeSize);
			}
			
			for (auto op : join) {
				op->launcher();
			}


		}
		

		//TODO modificare quando si sapra come utilizzare i risultati
		void printResults(google::dense_hash_map<size_t, std::string> mapH) {
			
			
			Operation* op;
			if (join.size() == 0) {
				std::cout << "LAST SELECT" << std::endl;
				op = select.back();
			} else {
				std::cout << "LAST JOIN" << std::endl;
				op = join.back();
			}
			
			std::cout << "FOUND " << op->getResult()->height << " ELEMENTS" << std::endl;
			
			Binding* d_result = op->getResult();
			std::vector<std::string> variables = op->getVariables();
				 	
			size_t* final_binding = (size_t*) malloc(d_result->height * d_result->width * sizeof(size_t));
			cudaMemcpy(final_binding, d_result->pointer, d_result->width * sizeof(size_t) * d_result->height, cudaMemcpyDeviceToHost);
	
			std::vector<std::string> output;
	
			for (int i =0; i < d_result->height; i++) {
				int currentVariable = 0;
				for (int k = 0; k < d_result->width; k++) {
					if (variables_projection[currentVariable] == variables[k]) {
						output.push_back(mapH[ final_binding[i * d_result->width + k]]);
						//std::cout << variables_projection[currentVariable] << "=" <<  mapH[ final_binding[i * d_result->width + k]] << " ";
						currentVariable++;
					}
				}
				//std::cout << std::endl;
			}	

			VALUE += op->getResult()->height;
			
			for (auto op : select) {
				delete(op->getResult());
			}
			
			for (auto op : join) {
				delete(op->getResult());
			}

			//TODO aggiungere clear della memoria dei risultati 		
		}
		
	
};


class CountQuery : public Query {
	private:
		int count;
		int currentCount;

	public:
		CountQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			CircularBuffer<TripleContainer> rdfPointer,
			std::vector<std::string> variables, int count) : Query(select, join, rdfPointer, variables) {

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
		CircularBuffer<unsigned long int> timestampPointer;
		//TIME IS IN M_SEC
		unsigned long int stepTime;
		unsigned long int windowTime;
		unsigned long int lastTimestamp;
		
	public:
		TimeQuery(std::vector<SelectOperation*> select, std::vector<JoinOperation*> join,
			CircularBuffer<TripleContainer> rdfPointer, CircularBuffer<unsigned long int> timestampPointer,
			std::vector<std::string> variables, int windowTime, int stepTime) : Query(select, join, rdfPointer, variables) {
				this->stepTime = stepTime;
				this->windowTime = windowTime;
				
				this->lastTimestamp = 0;
				this->timestampPointer = timestampPointer;
		}
		
		void setWindowEnd(int step)  {
			Query::setWindowEnd(step);
			timestampPointer.end = step;
		}

		bool isReady(unsigned long int newTimestamp) {
			return (lastTimestamp + windowTime < newTimestamp);
		}

		void setStartingTimestamp(unsigned long int timestamp) {
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


