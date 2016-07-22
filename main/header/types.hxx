#pragma once





//Struct to contains a single triple with int type.
struct tripleContainer {
        int subject;
        int predicate;
        int object;
};

//Struct
template<typename type_t>
struct bufferPointer {
	type_t* pointer;
	int begin;
	int end;
};


//Struct
template<typename rdf_t, typename arr_t>
struct devicePointer {
	rdf_t rdfStore;
	arr_t subject;
	arr_t predicate;
	arr_t object;
};
