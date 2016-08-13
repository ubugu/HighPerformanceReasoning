#pragma once

#include <cstdlib>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include "types.hxx"
#include "operations.hxx"
#include "rdfSelect.hxx"
#include "rdfJoin.hxx"


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

		std::vector<SelectOperation*> getSelect() {
			return select;
		}

		std::vector<JoinOperation*> getJoin() {
			return join;
		}

		virtual void advancePointer(int step) {
			windowPointer.advanceEnd(step);
		}

		triplePointer<tripleContainer*, int*> getStorePointer() {
			triplePointer<tripleContainer*, int*> pointer;
			pointer.rdfStore = windowPointer.rdfStore.pointer;
			pointer.subject = windowPointer.subject.pointer;
			pointer.predicate = windowPointer.predicate.pointer;
			pointer.object = windowPointer.object.pointer;
			
			return pointer;
		}
		
		void printResults() {
			int i = 0;
			for (auto op : select) {
				std::vector<tripleContainer> selectResults = from_mem(*(op->getResult()));
				std::cout <<"selct size " << selectResults.size() << std::endl;
				
				cudaFree(op->getResult()->data());
				
			/*	if (i <= 1) {
					TEST_VALUE[i] += selectResults.size();
				}*/
				
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
		
		~Query() {}
		
				/**
		* Function for managing query execution
		**/
		void startQuery() {
			int diff = windowPointer.rdfStore.end - windowPointer.rdfStore.begin;

			int storeSize = (diff >= 0 ? diff : windowPointer.rdfStore.size + diff); 
			
			std::vector<tripleContainer*> d_selectQueries;
			std::vector<int*> comparatorMask;
			std::vector<int> arrs;

			for (int i = 0; i < select.size(); i++) {
				d_selectQueries.push_back(select[i]->getQuery()->data());
				arrs.push_back(select[i]->getArr());
			}
	

			std::vector<mem_t<tripleContainer>*> selectResults = rdfSelect(d_selectQueries, windowPointer, storeSize, comparatorMask, arrs);

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

