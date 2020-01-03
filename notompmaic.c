//
//  main.c
//  AutoIO
//
//  Created by 王赫萌 on 2019/3/14.
//  Copyright © 2019 王赫萌. All rights reserved.
//

#include "common.h"
#include "mmio_highlevel.h"
#include "findlevel.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//#include <omp.h>

int main(int argc, const char * argv[]) {
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }
    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");
    int m, n, nnzA, isSymmetricA;
    int *csrRowPtr_tmp;
    int *csrColIdx_tmp;
    VALUE_TYPE *csrVal_tmp;
    int nnzTR;
    int *cscRowIdxTR;
    int *cscColPtrTR;
    VALUE_TYPE *cscValTR;
    int device_id = 0;
    int rhs = 0;
    int substitution = SUBSTITUTION_FORWARD;
    // "Usage: ``./sptrsv -d 0 -rhs 1 -forward -mtx A.mtx'' for LX=B on device 0"
    int argi = 1;
    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }
    if (strcmp(devstr, "-d") != 0) return 0;
    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    printf("device_id = %i\n", device_id);
    // load the number of right-hand-side
    char *rhsstr;
    if(argc > argi)
    {
        rhsstr = argv[argi];
        argi++;
    }

    if (strcmp(rhsstr, "-rhs") != 0) return 0;

    if(argc > argi)
    {
        rhs = atoi(argv[argi]);
        argi++;
    }
    printf("rhs = %i\n", rhs);

    // load substitution, forward or backward
    char *substitutionstr;
    if(argc > argi)
    {
        substitutionstr = argv[argi];
        argi++;
    }

    if (strcmp(substitutionstr, "-forward") == 0)
        substitution = SUBSTITUTION_FORWARD;
    else if (strcmp(substitutionstr, "-backward") == 0)
        substitution = SUBSTITUTION_BACKWARD;
    printf("substitutionstr = %s\n", substitutionstr);
    printf("substitution = %i\n", substitution);
    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    printf("matstr = %s\n", matstr);
    // load matrix data from file
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("-------------- %s --------------\n", filename);
    srand(time(NULL));
  // load mtx data to the csr format
    mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
    csrRowPtr_tmp = (int *)malloc((m+1) * sizeof(int));
    csrColIdx_tmp = (int *)malloc(nnzA * sizeof(int));
    csrVal_tmp    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    mmio_data(csrRowPtr_tmp, csrColIdx_tmp, csrVal_tmp, filename);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
        // extract L or U with a unit diagonal of A
        int *csrRowPtr = (int *)malloc((m+1) * sizeof(int));
        int *csrColIdx = (int *)malloc((m+nnzA) * sizeof(int));
        VALUE_TYPE *csrVal    = (VALUE_TYPE *)malloc((m+nnzA) * sizeof(VALUE_TYPE));
        int nnz_pointer = 0;
        csrRowPtr[0] = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtr_tmp[i]; j < csrRowPtr_tmp[i+1]; j++)
            {  
                if (substitution == SUBSTITUTION_FORWARD)
                {
                    if (csrColIdx_tmp[j] < i)
                    {
                        csrColIdx[nnz_pointer] = csrColIdx_tmp[j];
                        csrVal[nnz_pointer] = 1;//rand() % 10 + 1; //csrVal_tmp[j];
                        nnz_pointer++;
                    }
                }
                else if (substitution == SUBSTITUTION_BACKWARD)
                {
                    if (csrColIdx_tmp[j] > i)
                    {
                        csrColIdx[nnz_pointer] = csrColIdx_tmp[j];
                        csrVal[nnz_pointer] = 1;//rand() % 10 + 1; //csrVal_tmp[j];
                        nnz_pointer++;
                    }
                }
            }
            // add dia nonzero
            csrColIdx[nnz_pointer] = i;
            csrVal[nnz_pointer] = 1.0;
            nnz_pointer++;
            csrRowPtr[i+1] = nnz_pointer;
        }
        nnzTR = csrRowPtr[m];
        if (substitution == SUBSTITUTION_FORWARD)
            printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        else if (substitution == SUBSTITUTION_BACKWARD)
            printf("A's unit-upper triangular U: ( %i, %i ) nnz = %i\n", m, n, nnzTR);
        csrColIdx = (int *)realloc(csrColIdx, sizeof(int) * nnzTR);
        csrVal = (VALUE_TYPE *)realloc(csrVal, sizeof(VALUE_TYPE) * nnzTR);
        cscRowIdxTR = (int *)malloc(nnzTR * sizeof(int));
        cscColPtrTR = (int *)malloc((n+1) * sizeof(int));
        memset(cscColPtrTR, 0, (n+1) * sizeof(int));
        cscValTR    = (VALUE_TYPE *)malloc(nnzTR * sizeof(VALUE_TYPE));
        // transpose from csr to csc
        matrix_transposition(m, n, nnzTR,
                             csrRowPtr, csrColIdx, csrVal,
                             cscRowIdxTR, cscColPtrTR, cscValTR);
        // keep each column sort
    int nlevel = 0;
    int parallelism_min = 0;
    int parallelism_avg = 0;
    int parallelism_max = 0;
    int  *levelPtr  = (int *)malloc((m+1) * sizeof(int));
    int  *levelItem = (int *)malloc(m * sizeof(int));
    findlevel_csr(csrRowPtr, csrColIdx, csrVal, m, n, nnzTR, &nlevel,
                  &parallelism_min, &parallelism_avg, &parallelism_max,
                  levelPtr,levelItem);
    // find level sets
    //findlevel_csc(cscColPtrTR, cscRowIdxTR, cscValTR, m, n, nnzTR, &nlevel,
    //                &parallelism_min, &parallelism_avg, &parallelism_max);
    double fparallelism = (double)m/(double)nlevel;
    printf("This matrix/graph has %i levels, its parallelism is %4.2f (min: %i ; avg: %i ; max: %i )\n",
           nlevel, fparallelism, parallelism_min, parallelism_avg, parallelism_max);
    double *results = (double *)malloc(m * sizeof(double));//the results

    int counter = 0;//to prevent the same elem
    double s = 0;//temperately memory the past sum

    VALUE_TYPE *b = (double *)malloc(m * sizeof(double));//AX=b
    //printf("Input the bs:\n");
    srand(time(NULL));
    for (int i = 0 ; i < m ; i++) {
        b[i] = 1;//rand()%1800;
        //scanf("%lf",&b[i]); getchar();
    }

    clock_t start,finish; double TheTimes;
    start = clock();//start to clock

    for(int loop = 0; loop < 100; loop++)
    {//printf("%d\n",loop);
        for (int k = 0; k < nlevel; k++) {//the kth level
        //#pragma omp parallel for
            for (int j = levelPtr[k] ; j < levelPtr[k+1]; j++) {//parallel the level k
               int i = levelItem[j];//the row need be solved
               int s = 0;
               int now = csrRowPtr[i];
                while(now < csrRowPtr[i+1]-1){
                    s += (csrVal[now] * results[csrColIdx[now]]);
                   now++;
                }
                results[i] = (b[i] - s) / csrVal[now];
            }
        }
    }
    
    


    finish = clock();//end the clock
    TheTimes = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("%f seconds。\n",TheTimes/100);
    /*for (int i = 0 ; i < m ; i++) {
        printf("x%d : %lf\n",i+1,results[i]);
    }printf("\n");
    for (int i = 0 ; i < nnzTR ; i++) {
        printf("%lf ",csrVal[i]);
    } printf("\n");
    for (int i = 0 ; i < nnzTR ; i++) {
        printf("%d ",csrColIdx[i]);
    } printf("\n");
    for (int i = 0 ; i <= m ; i++) {
        printf("%d ",csrRowPtr[i]);
    }printf("\n");
    for (int i = 0 ; i < m ; i++) {
        printf("%f ",b[i]);
    }printf("\n");
    for (int i = 0 ; i < m ; i++) {
        printf("%d ",levelItem[i]);
    }printf("\n");
    for (int i = 0 ; i < 4 ; i++) {
        printf("%d ",levelPtr[i]);
    }printf("\n");*/
    free(levelItem);
    free(levelPtr);
    free(csrColIdx);
    free(csrVal);
    free(csrRowPtr);
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowPtr_tmp);
    free(cscRowIdxTR);
    free(cscColPtrTR);
    free(cscValTR);
    free(results);
    free(b);
    return 0;
}
/*
s = 0;
            int i = levelItem[j];
            int l = csrRowPtr[i];//to calculate the loop
            while (l < csrRowPtr[i+1] - 1) {
                s += (csrVal[counter] * results[csrColIdx[counter]]);
                //use matrix.col[counter] to find the results
                counter++;//after sum, the elem is passed
                l++;
            }
            results[i] = (b[i] - s) / csrVal[counter];
            counter++;//after division, the elem is passed
*/
/*


    int elem_temp = 0;//locate the present elem
    int elem = 0;
    csrRowPtr[0] = 0;
    for (int i = 0 ; i < m ; i++) {
        int j = 0;
        for (j = 0 ; j < csrRowElem_tmp[i]; j++) {
            if (csrColIdx_tmp[elem_temp] < i) {
                csrVal[elem] = 1;//csrVal_tmp[elem_temp];//to prvent the double explod
                csrColIdx[elem] = csrColIdx_tmp[elem_temp];
                elem++;
            }
            elem_temp++;
        }
        csrVal[elem] = 1;
        csrColIdx[elem] = i;
        elem++;
        csrRowPtr[i+1] = elem;
    }
    nnz = elem;
    printf("nnz:%d\n",nnz);
*/
