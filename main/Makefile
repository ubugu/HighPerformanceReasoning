all:
	nvcc -std=c++11 --ptxas-options=-v -O4 -Xcompiler -D_FORCE_INLINES --expt-extended-lambda -gencode arch=compute_61,code=compute_61 -D_MWAITXINTRIN_H_INCLUDED -I ../moderngpu/src/ queryTest.cu  -o timeHash

