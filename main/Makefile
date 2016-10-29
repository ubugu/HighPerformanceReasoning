all:
	nvcc -std=c++11 --ptxas-options=-v -O4 -Xcompiler -D_FORCE_INLINES --expt-extended-lambda -gencode arch=compute_35,code=compute_35 -D_MWAITXINTRIN_H_INCLUDED -I ../moderngpu/src/ -I ./sparsehash-master/src/ queryTimeHashOr.cu  -o timeHash

