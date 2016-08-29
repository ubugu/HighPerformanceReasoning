#pragma once

//TODO 
//VARIABILI PER TESTING, DA RIMUOVERE DAL CODICE FINALE
int VALUE = 0;
std::vector<float> timeCuVector;                
std::vector<long int> timeExVector;
//**END TESTING***//


//struct to contains a single triple with int type.
struct TripleContainer {
        size_t subject;
        size_t predicate;
        size_t object;

	void print() {
		std::cout << subject << " " << predicate << " " << object << std::endl;
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

struct Binding {
	size_t* pointer;
	int width;
	int height;
	std::vector<std::string> table_header;
	
	Binding() {}
	
	Binding(int width, int height) {
		cudaMalloc(&pointer, width * height *  sizeof(size_t));
		this->width = width;
		this->height = height;
	}
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

const basic_fnv_1< 1099511628211u, 14695981039346656037u> h_func;



