#pragma once

#include <moderngpu/context.hxx>

//TODO VARIABILI PER TESTING, DA RIMUOVERE DAL CODICE FINALE
int VALUE = 0;
int TEST = 0;
std::vector<float> timeQueryVector;  
std::vector<float> timeKernelVector;  
std::vector<float> timeStoreVector;
std::vector<float> hashVector;
std::vector<float> testVector;  
//**END TESTING***//

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


//NUMERIC contains: FLOAT, INTEGER, DOUBLE
enum class Datatype{ STR = 0,  NUMERIC = 1, DATETIME = 2, BOOL = 3, URI = 4};

//37 basic length
//Considering type like "%value%"^^http://www.w3.org/2001/XMLSchema#%datatype%
struct Lit  {
	Datatype type;

	double numericValue = 0;
	
	std::string stringValue;
	
	Lit(){}
	Lit(double val, Datatype typ, std::string stringVal) : stringValue(stringVal), numericValue(val), type(typ){}
	Lit(std::string stringVal) : stringValue(stringVal), type(Datatype::STR){}
	Lit(std::string stringVal, Datatype typ) : stringValue(stringVal), type(typ){}
	
	//Function for creating literal, deducing the type from the string
	static Lit createLiteral(std::string literalStr) {
		int length = literalStr.length();
		
		if (literalStr[0] == '<') {
			Lit literal(literalStr, Datatype::URI);
			return literal;
		}
	 
		if (length < 42) {
			Lit literal(literalStr);
			return literal;
		}
		
		if  (literalStr[length - 1] == 'g') {
			Lit literal(literalStr.substr(0, length - 42));
			return literal;
		}

		if (literalStr[length - 1] == 'r') {
			std::string stringValue = literalStr.substr(1, length - 44);
			double value = stod (stringValue);
			Lit literal(value, Datatype::NUMERIC, literalStr);
			return literal;
		}

	
		if (literalStr[length - 1] == 't') {
			std::string stringValue = literalStr.substr(1, length - 42);
			double value = stod (stringValue);
			Lit literal(value, Datatype::NUMERIC, literalStr);
			return literal;
		}	
	
	
		if (literalStr[length - 2] == 'l') {
			std::string stringValue = literalStr.substr(1, length - 43);
			double value = stod (stringValue);
			Lit literal(value, Datatype::NUMERIC, literalStr);
			return literal;
		}
		
		std::cout << "ERROR RESOURCE ENCOUNTERED" << std::endl;
		exit(-1);
	
	}

};

//Struct for circular buffer
template<typename type_t>
struct CircularBuffer {
	type_t* pointer;
	int begin;
	int end;
	int size;
	
	CircularBuffer() : pointer(0), begin(0), end(0), size(0) {}
	
	int getLength() {
		return (abs(end - begin + size) % size);
	}
	
	void advanceBegin(int step){
		begin = (begin + step) % size;
	}	
};

//Struct for relational table
struct RelationTable {
	size_t* pointer;
	int width;
	int height;
	std::vector<std::string> header;
	bool onDevice = false;
	
	RelationTable() {}

	//Allocate table on device
	void allocateOnDevice(int height) {
		onDevice = true;
		width = header.size();
		this->height = height;
		cudaMalloc(&pointer, height * width * sizeof(size_t));	
	}
	
	~RelationTable(){}
};



/**
**
** CODE FOR HASH FUNCTION
**
**/
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

const basic_fnv_1< 1099511628211u, 14695981039346656037u> hashFunction;



uint32_t sdbm (const char * data, int len) {
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





