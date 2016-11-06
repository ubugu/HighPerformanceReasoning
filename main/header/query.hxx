#pragma once

#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include <sparsehash/dense_hash_map>

#include "types.hxx"
#include "operations.hxx"
#include "outputDirective.hxx"

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


class CountQuery : public Query {
	private:
		int count_;
		int currentCount_;

	public:
		CountQuery(std::vector<Operation*> operations, CircularBuffer<size_t>* rdfPointer, OutputDirective* directive, int count) : Query(operations, rdfPointer, directive) {

				this->count_ = count;
				this->currentCount_ = 0;
		}
		
		void incrementCount() {
			this->currentCount_++;
		}
		
		bool isReady() {
			return (currentCount_ == count_);
		}
		
		void launch() {
			startQuery();
			windowPointer_->advanceBegin(count_);
			currentCount_ = 0;
		}
		
		~CountQuery() {}
};

class TimeQuery : public Query {
	private:
		CircularBuffer<unsigned long int> timestampPointer_;
		
		//TIME IS IN ms_SEC
		unsigned long int stepTime_;
		unsigned long int windowTime_;
		unsigned long int lastTimestamp_;
		
	public:
		TimeQuery(std::vector<Operation*> operations, CircularBuffer<size_t>* rdfPointer, CircularBuffer<unsigned long int> timestampPointer,
			 OutputDirective* directive, int windowTime, int stepTime) : Query(operations, rdfPointer, directive) {
				this->stepTime_ = stepTime;
				this->windowTime_ = windowTime;
				
				this->lastTimestamp_ = 0;
				this->timestampPointer_ = timestampPointer;
		}
		
		void setWindowEnd(int step)  {
			Query::setWindowEnd(step);
			timestampPointer_.end = step;
		}

		bool isReady(unsigned long int newTimestamp) {
			return (lastTimestamp_ + windowTime_ < newTimestamp);
		}

		void setStartingTimestamp(unsigned long int timestamp) {
			this->lastTimestamp_ = timestamp;
		}

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
			struct timeval beginK, endK;
			gettimeofday(&beginK, NULL);	
			
			startQuery();

			cudaDeviceSynchronize();
			gettimeofday(&endK, NULL);
			float KTime = (endK.tv_sec - beginK.tv_sec ) * 1000 + ((float) endK.tv_usec - (float) beginK.tv_usec) / 1000 ;
			timeKernelVector.push_back(KTime);

			//Update window timestamp value
			lastTimestamp_ += stepTime_;
		}

		~TimeQuery() {}
};


