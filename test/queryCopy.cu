#include <iostream>
#include <fstream>
#include <cstdlib>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_join.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <sys/time.h>
#include <functional>

using namespace mgpu;
.
template<typename element_t>
struct tripleContainer {
        element_t subject;
        element_t predicate;
        element_t object;
};


		struct timeval beginPr, beginCu, beginEx, end;
		gettimeofday(&beginPr, NULL);	
		
		cudaDeviceReset();
		standard_context_t context;
                const int FILE_LENGHT = 100000;
                size_t rdfSize = FILE_LENGHT * sizeof(tripleContainer<int>);
                tripleContainer<int>* h_rdfStore = (tripleContainer<int>*) malloc(rdfSize);

                //read store from rdfStore
                ifstream rdfStoreFile ("../utilities/rdf30k-3k.txt");

                string strInput;

                for (int i = 0; i < FILE_LENGHT; i++) {
                        getline(rdfStoreFile,strInput);

                        std::vector<string> triple ;
                        separateWords(strInput, triple, ' ');
			
			h_rdfStore[i].subject = atoi(triple[0].c_str());
			h_rdfStore[i].predicate = atoi(triple[1].c_str());
			h_rdfStore[i].object = atoi(triple[2].c_str());
                }
                rdfStoreFile.close();

		gettimeofday(&beginCu, NULL);

		tripleContainer<int>* d_storeVector;
		cudaMalloc(&d_storeVector, rdfSize);
		cudaMemcpy(d_storeVector, h_rdfStore, rdfSize, cudaMemcpyHostToDevice);	
			
                //set Queries (select that will be joined)
                tripleContainer<int> h_queryVector1 { 0 , 200 , 2 }; 
                tripleContainer<int> h_queryVector2 { 0 , 200 , 2 };
                
                mem_t<tripleContainer<int>> d_queryVector1(1, context);
		cudaMemcpy(d_queryVector1.data(), &h_queryVector1, sizeof(tripleContainer<int>), cudaMemcpyHostToDevice);
		
                mem_t<tripleContainer<int>> d_queryVector2(1, context);
		cudaMemcpy(d_queryVector2.data(), &h_queryVector2, sizeof(tripleContainer<int>), cudaMemcpyHostToDevice);
		
		//set select mask operation
		std::vector<tripleContainer<int>*> selectQuery;
		selectQuery.push_back(d_queryVector1.data());
		selectQuery.push_back(d_queryVector2.data());

		std::vector<compareType*> compareMask;
		compareType selectMask1[3];
		
		selectMask1[0] = compareType::EQ;
		selectMask1[1] = compareType::NC;
		selectMask1[2] = compareType::NC;

		compareMask.push_back(selectMask1);
		
		compareType selectMask2[3];		
		selectMask2[0] = compareType::EQ;
		selectMask2[1] = compareType::NC;
		selectMask2[2] = compareType::NC;
		
		compareMask.push_back(selectMask2);
		
		//set Join mask
		int joinMask[3];
		joinMask[0] = 1;
		joinMask[1] = 0;
		joinMask[2] = 0;
			
		SelectOperation<int>  selectOp1(&d_queryVector1, selectMask1);
		SelectOperation<int>  selectOp2(&d_queryVector2, selectMask2);
		
		JoinOperation<int>  joinOp(selectOp1.getResultAddress(), selectOp2.getResultAddress(), joinMask);
		
		std::vector<SelectOperation<int>*> selectOperations;
		std::vector<JoinOperation<int>*> joinOperations;
		
		selectOperations.push_back(&selectOp1);
		selectOperations.push_back(&selectOp2);
		joinOperations.push_back(&joinOp);
		
		gettimeofday(&beginEx, NULL);	
	

				
		queryManager<int>(selectOperations, joinOperations, d_storeVector, FILE_LENGHT);
		cudaDeviceSynchronize();
		
		gettimeofday(&end, NULL);
		
		float exTime = (end.tv_sec - beginEx.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginEx.tv_usec) / 1000 ;
		float prTime = (end.tv_sec - beginPr.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginPr.tv_usec) / 1000 ;
		float cuTime = (end.tv_sec - beginCu.tv_sec ) * 1000 + ((float) end.tv_usec - (float) beginCu.tv_usec) / 1000 ;
		
		cout << "Total time: " << prTime << endl;
		cout << "Cuda time: " << cuTime << endl;
		cout << "Execution time: " << exTime << endl;
		
		cout << "first select result" << endl;
		std::vector<tripleContainer<int>> selectResults = from_mem(*selectOp1.getResult());
		cout << selectResults.size() << endl;
	/*	for (int i = 0; i < selectResults.size(); i++) {
			cout << selectResults[i].subject << " " << selectResults[i].predicate << " "  << selectResults[i].object << endl; 
		}
	*/	
		cout << "second select result" << endl;
		std::vector<tripleContainer<int>> selectResults2 = from_mem(*selectOp2.getResult());
		cout << selectResults2.size() << endl;
/*		for (int i = 0; i < selectResults2.size(); i++) {
			cout << selectResults2[i].subject << " " << selectResults2[i].predicate << " "  << selectResults2[i].object << endl; 
		}
*/		
		cout << "final result" << endl;
		std::vector<tripleContainer<int>> finalResults = from_mem(*joinOp.getResult());
		cout << finalResults.size() << endl;
	/*	for (int i = 0; i < finalResults.size(); i++) {
			cout << finalResults[i].subject << " " << finalResults[i].predicate << " "  << finalResults[i].object << endl; 
		}
	*/	
		return 0;

}



