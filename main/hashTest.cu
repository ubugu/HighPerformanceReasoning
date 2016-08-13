#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include<vector>
#include <chrono>
#include <string>
#include <unordered_map>
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>

#include <time.h>

#include <sparsehash/sparse_hash_map>

using google::sparse_hash_map; 

#include "stdint.h" 
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif

uint32_t SuperFastHash (const char * data, int len) {
uint32_t hash = len, tmp;
int rem;

    if (len <= 0 || data == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

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

template <std::size_t FnvPrime, std::size_t OffsetBasis>
struct basic_fnv_1a
{
    std::size_t operator()(std::string const& text) const
    {
            std::size_t hash = OffsetBasis;
            for(std::string::const_iterator it = text.begin(), end = text.end();
                    it != end; ++it)
            {
                hash ^= *it;
                hash *= FnvPrime;
            }

            return hash;
    }
};
    
    
    unsigned long
djb2(const char *str)
{
    unsigned long hash = 5381;
    int c;

    while (c = *str++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

//struct to contains a single triple with int type.
template<typename type_t>
struct tripleContainer {
        type_t subject;
        type_t predicate;
        type_t object;
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

struct eqstr
{
  bool operator()(size_t s1, size_t s2) const
  {
    return (s1 == s2);
   }
};


struct eqstr2
{
  bool operator()(const char* s1, const char* s2) const
  {
    return (s1 == s2) || (s1 && s2 && strcmp(s1, s2) == 0);
  }
};


using google::dense_hash_map;      // namespace where class lives by default
using std::cout;
using std::endl;



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
		
		ifstream rdfStoreFile (argv[1]);
		string strInput;
		int fileLength = 0;	 
		while (std::getline(rdfStoreFile, strInput)) {
			++fileLength;
		}
	
		rdfStoreFile.clear();
		rdfStoreFile.seekg(0, ios::beg);
		int LEN = fileLength;
                size_t rdfSize = fileLength  * sizeof(tripleContainer<std::string>);
                tripleContainer<std::string*>* h_rdfStore = (tripleContainer<std::string*>*)  malloc(rdfSize);


		std::cout << "FILE LENGHT IS " << fileLength << std::endl;

                //read store from rdfStore
                for (int i = 0; i <fileLength; i++) {
			getline(rdfStoreFile,strInput);
                        std::vector<string> triple;
                        separateWords(strInput, triple, ' ');                       
                        h_rdfStore[i].subject = new std::string(triple[0]);
                        h_rdfStore[i].predicate = new std::string(triple[1]);
                        h_rdfStore[i].object = new std::string(triple[2]);
                }
                rdfStoreFile.close();
          				
                size_t rdfSize2 = fileLength  * sizeof(tripleContainer<const char*>);
                tripleContainer<const char*>* h_rdfStore2 = (tripleContainer<const char*>*)  malloc(rdfSize2);
                for (int i = 0; i <fileLength; i++) {             
                        h_rdfStore2[i].subject =  h_rdfStore[i].subject->c_str();
                        h_rdfStore2[i].predicate =  h_rdfStore[i].subject->c_str(); 
                        h_rdfStore2[i].object =  h_rdfStore[i].subject->c_str(); 
                }
          	
          	
          	
          	
          	
          	
          	
                tripleContainer<size_t> results;
          
 
		unordered_map<size_t, std::string> hashtable;

  		dense_hash_map<size_t, const char*, hash<size_t>, eqstr> months;
  
  		months.set_empty_key(NULL);          
          
          	struct timeval begin,end;
          	float exTime = 0;
                std::vector<float> time;		
		std::vector<float> statistics;
		
		
		        
          	//CALCULATE EXECUTION TIME
          	
          	//FNV 1
          	for (int k = 0; k < 100; k++) {
          	gettimeofday(&begin, NULL);	
   		basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;      
		for (int i =0; i <LEN; i++) {	
			months[h_func(*h_rdfStore[i].subject)] = h_rdfStore[i].subject->c_str();
			months[h_func(*h_rdfStore[i].predicate)] = h_rdfStore[i].predicate->c_str();	
			months[h_func(*h_rdfStore[i].object)] = h_rdfStore[i].object->c_str();				
		}
		gettimeofday(&end, NULL);
		exTime = (end.tv_sec - begin.tv_sec ) * 1000 + ((float) end.tv_usec - (float) begin.tv_usec) / 1000 ;
		time.push_back(exTime);
		
		}
		statistics = stats<float, float>(time);	
                cout << "FNV 1 mean cuda time " << statistics[0] << "ms"<< endl;
                cout << "FNV 1 variance cuda time " << statistics[1] << endl;
		time.clear();
		months.clear();

/*
          	//FNV 1
          	for (int k = 0; k < 100; k++) {
          	gettimeofday(&begin, NULL);	
   		basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;      
		for (int i =0; i <LEN; i++) {	
			hashtable.insert(std::make_pair( h_func(*h_rdfStore[i].subject), *h_rdfStore[i].subject));
			hashtable.insert(std::make_pair( h_func(*h_rdfStore[i].predicate), *h_rdfStore[i].predicate));
			hashtable.insert(std::make_pair( h_func(*h_rdfStore[i].object),  *h_rdfStore[i].object)) ;		
		}
		gettimeofday(&end, NULL);
		exTime = (end.tv_sec - begin.tv_sec ) * 1000 + ((float) end.tv_usec - (float) begin.tv_usec) / 1000 ;
		time.push_back(exTime);
		
		}
		statistics = stats<float, float>(time);	
                cout << "FNV 1 mean cuda time " << statistics[0] << "ms"<< endl;
                cout << "FNV 1 variance cuda time " << statistics[1] << endl;
		time.clear();
		months.clear();

          	//FNV 1a
          	for (int k = 0; k < 100; k++) {
          	gettimeofday(&begin, NULL);	
   		basic_fnv_1a< 1099511628211u, 14695981039346656037u> h_funca;      
		for (int i =0; i <LEN; i++) {	
			months[h_funca(*h_rdfStore[i].subject)] = h_rdfStore[i].subject->c_str();
			months[h_funca(*h_rdfStore[i].predicate)] = h_rdfStore[i].predicate->c_str();	
			months[h_funca(*h_rdfStore[i].object)] = h_rdfStore[i].object->c_str();				
		}
		gettimeofday(&end, NULL);
		exTime = (end.tv_sec - begin.tv_sec ) * 1000 + ((float) end.tv_usec - (float) begin.tv_usec) / 1000 ;
		time.push_back(exTime);
		
		}
		statistics = stats<float, float>(time);	
                cout << "FNV 1a mean cuda time " << statistics[0] << "ms"<< endl;
                cout << "FNV 1a variance cuda time " << statistics[1] << endl;
		time.clear();
		months.clear();				
		
		
		//Super Fast Hash
          	for (int k = 0; k < 100; k++) {
          	gettimeofday(&begin, NULL);	
		for (int i =0; i <LEN; i++) {
			months[SuperFastHash ( h_rdfStore[i].subject->c_str(),  h_rdfStore[i].subject->length())] = h_rdfStore[i].subject->c_str();
			months[SuperFastHash(h_rdfStore[i].predicate->c_str(),  h_rdfStore[i].predicate->length())] = h_rdfStore[i].predicate->c_str();	
			months[SuperFastHash(h_rdfStore[ i].object->c_str(),  h_rdfStore[i].object->length())] = h_rdfStore[i].object->c_str();			
		}
		gettimeofday(&end, NULL);
		exTime = (end.tv_sec - begin.tv_sec ) * 1000 + ((float) end.tv_usec - (float) begin.tv_usec) / 1000 ;
		time.push_back(exTime);
		}
		statistics = stats<float, float>(time);	
                cout << "SFH mean cuda time " << statistics[0] << "ms"<< endl;
                cout << "SFH variance cuda time " << statistics[1] << endl;
		time.clear();
		months.clear();


          	//DJB2
          	for (int k = 0; k < 100; k++) {
          	gettimeofday(&begin, NULL);	
   		basic_fnv_1a< 1099511628211u, 14695981039346656037u> h_func;      
		for (int i =0; i <LEN; i++) {	
			months[djb2(h_rdfStore2[i].subject)] = h_rdfStore[i].subject->c_str();
			months[djb2(h_rdfStore2[i].predicate)] = h_rdfStore[i].predicate->c_str();	
			months[djb2(h_rdfStore2[i].object)] = h_rdfStore[i].object->c_str();				
		}
		gettimeofday(&end, NULL);
		exTime = (end.tv_sec - begin.tv_sec ) * 1000 + ((float) end.tv_usec - (float) begin.tv_usec) / 1000 ;
		time.push_back(exTime);
		
		}
		statistics = stats<float, float>(time);	
                cout << "DJB2 1 mean cuda time " << statistics[0] << "ms"<< endl;
                cout << "DJB2 1 variance cuda time " << statistics[1] << endl;
		time.clear();
		months.clear();
*/


		//CALCULATE COLLISIONS 
		int collisions = 0;
		hashtable.clear();
		
			using namespace std;
	
		std::vector<std::string> h_rdfStore3;
		std::vector<size_t> hashed;
		gettimeofday(&end, NULL);
		srand(end.tv_usec);
		
		int sizze = 1000000;
		
		while (h_rdfStore3.size() < sizze) {
			
			std::string stringa = "<http://example.org/int/";
			
			for (int i = 0; i < 7; i++) {
				int random = 97 + rand() % (122 - 97);
				stringa += static_cast<char>(random);
			}
			
			//std::cout << stringa << std::endl;
			h_rdfStore3.push_back(stringa);
		}
		std::sort(h_rdfStore3.begin(), h_rdfStore3.end()); 
  		auto last = std::unique(h_rdfStore3.begin(), h_rdfStore3.end());
   		h_rdfStore3.erase(last, h_rdfStore3.end());  	
		std::random_shuffle ( h_rdfStore3.begin(), h_rdfStore3.end() );	
		std::cout << "size is " << h_rdfStore3.size() << std::endl;
		size_t oldSize = 0;
		
		

		
		
		
		
		//FNV 1
		basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;   
		for (int i =0; i <h_rdfStore3.size(); i++) {
			
			hashed.push_back( h_func(h_rdfStore3[i]) );
			

		}
		oldSize = hashed.size();
		std::sort(hashed.begin(), hashed.end()); 
   		hashed.erase(std::unique(hashed.begin(), hashed.end()), hashed.end()); 	
		std::random_shuffle ( hashed.begin(), hashed.end() );	
		std::cout << "SIZE IS " << hashed.size() << std::endl;
		std::cout << "old is " << oldSize << std::endl;
		std::cout << "FNV 1 collisions are " << hashed.size() - oldSize << std::endl;
		collisions = 0;

		
		

		
		hashed.clear();
		
		//FNV 1
		basic_fnv_1a< 1099511628211u, 14695981039346656037u> h_funca;   
		for (int i =0; i <h_rdfStore3.size(); i++) {
			
			hashed.push_back( h_funca(h_rdfStore3[i]) );
			

		}
		oldSize = hashed.size();
		std::sort(hashed.begin(), hashed.end()); 
   		hashed.erase(std::unique(hashed.begin(), hashed.end()), hashed.end()); 	
		std::random_shuffle ( hashed.begin(), hashed.end() );	
		std::cout << "SIZE IS " << hashed.size() << std::endl;
		std::cout << "old is " << oldSize << std::endl;	
		std::cout << "FNV 1a collisions are " << hashed.size() - oldSize << std::endl;
		hashed.clear();
		
		
		std::vector<uint32_t> hashed2;
		uint32_t oldSize2;
		//FNV 1  
		for (int i =0; i <h_rdfStore3.size(); i++) {
			
			hashed2.push_back(SuperFastHash ( h_rdfStore3[i].c_str(),  h_rdfStore3[i].length()) );
			

		}
		oldSize2 = hashed2.size();
		std::sort(hashed2.begin(), hashed2.end()); 
   		hashed2.erase(std::unique(hashed2.begin(), hashed2.end()), hashed2.end()); 	
		std::random_shuffle ( hashed2.begin(), hashed2.end() );
		std::cout << "SIZE IS " << hashed2.size() << std::endl;
		std::cout << "old is " << oldSize << std::endl;
		std::cout << "SFH 1 collisions are " << oldSize2 - hashed2.size()   << std::endl;
		collisions = 0;
		hashed2.clear();	
		
		
		
		
		
		
		
		
                return 0;
}
