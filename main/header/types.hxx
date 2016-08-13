#pragma once

//Struct to contains a single triple with int type.
struct tripleContainer {
        int subject;
        int predicate;
        int object;
};

template<typename rdf_t, typename arr_t>
struct triplePointer {
	rdf_t rdfStore;
	arr_t subject;
	arr_t predicate;
	arr_t object;
};

template<typename type_t>
struct circularBuffer {
	type_t* pointer;
	int begin;
	int end;
	int size;
	
	circularBuffer() : pointer(0), begin(0), end(0), size(0) {}
	
	circularBuffer(int begin, int size, type_t* pointer) {
		this->begin = begin;
		this->end = begin;
		this->size = size;
		this->pointer = pointer;
	}

};



struct deviceCircularBuffer : triplePointer<circularBuffer<tripleContainer>, circularBuffer<int>> {
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
	
	void advanceEnd(int step){
		int newEnd = (rdfStore.end + step) % rdfStore.size;
		setEnd(newEnd);
	}
	
				
};


