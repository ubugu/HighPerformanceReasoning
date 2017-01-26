#pragma once

#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include "types.hxx"
#include "operations.hxx"
#include "outputDirective.hxx"

//Base class for query
class Query {
	protected:
		std::vector<Operation*> operations_;
		CircularBuffer<size_t>* windowPointer_;
		OutputDirective* outputDirective_;
	public:
		Query(std::vector<Operation*> operations, CircularBuffer<size_t>* rdfPointer,  OutputDirective* directive) {
			this->windowPointer_ = rdfPointer;
			this->operations_ = operations;
			outputDirective_ =  directive;
		}

		virtual void setWindowEnd(int step) {
			windowPointer_->end = step;
		}
				
		//Function for managing query execution
		void startQuery() {
			for (auto op : operations_) {
				op->execute();
			}
		}
		
		//Function for generating the output
		void printResults(std::unordered_map<size_t, Lit> mapH) {
			RelationTable d_result = operations_.back()->getResult();
			outputDirective_->generateOutput(mapH, d_result);
			
			//Free result memory
			if (d_result.onDevice) {
				cudaFree(d_result.pointer);
			} else {
				free(d_result.pointer);
			}
					
		}
			
};

//Class for physical query
class CountQuery : public Query {
	private:
		int count_;
		int currentCount_;

	public:
		CountQuery(std::vector<Operation*> operations, CircularBuffer<size_t>* rdfPointer, OutputDirective* directive, unsigned long int count) : Query(operations, rdfPointer, directive) {

				this->count_ = count;
				this->currentCount_ = 0;
		}
		
		//Increment current counter of incoming tryple
		void incrementCount() {
			this->currentCount_++;
		}
		
		//Check if the query is ready
		bool isReady() {
			return (currentCount_ == count_);
		}
		
		
		//Launche the query
		void launch() {
			startQuery();
			windowPointer_->advanceBegin(count_);
			currentCount_ = 0;
		}
		
		~CountQuery() {}
};

//Class for logical query
class TimeQuery : public Query {
	private:
		CircularBuffer<unsigned long int> timestampPointer_;
		
		//Time is expressed in ms
		unsigned long int stepTime_;
		unsigned long int windowTime_;
		unsigned long int lastTimestamp_;
		
	public:
		TimeQuery(std::vector<Operation*> operations, CircularBuffer<size_t>* rdfPointer, CircularBuffer<unsigned long int> timestampPointer,
			 OutputDirective* directive, int windowTime, unsigned long int stepTime) : Query(operations, rdfPointer, directive) {
				this->stepTime_ = stepTime;
				this->windowTime_ = windowTime;
				
				this->lastTimestamp_ = 0;
				this->timestampPointer_ = timestampPointer;
		}
		
		//Update pointer to the store timestamps
		void setWindowEnd(int step)  {
			Query::setWindowEnd(step);
			timestampPointer_.end = step;
		}

		//Check if the query is readi
		bool isReady(unsigned long int newTimestamp) {
			return (lastTimestamp_ + windowTime_ < newTimestamp);
		}

		//Set starting timestamp
		void setStartingTimestamp(unsigned long int timestamp) {
			this->lastTimestamp_ = timestamp;
		}

		//Start query
		void launch() {	
			//Update new starting value of buffer
			int newBegin = 0;
			bool isFirst = true;

			for(int i = timestampPointer_.begin; (i  != timestampPointer_.end) || (isFirst); i = (i + 1) % timestampPointer_.size) {	
				if (timestampPointer_.pointer[i] > lastTimestamp_) {
					newBegin = i;
					break;
				}
				isFirst = false;			
			}
			
			windowPointer_->begin = newBegin;
			timestampPointer_.begin = newBegin;
			
   
			//Lancuh query and print results
			startQuery();



			//Update window timestamp value
			lastTimestamp_ += stepTime_;
		}

		~TimeQuery() {}
};


