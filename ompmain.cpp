//  main.c
//  AutoIO
//
//  Created by 王赫萌 on 2019/3/14.
//  Copyright © 2019 王赫萌. All rights reserved.
//

#include "common.h"
#include "mmio_highlevel.h"
#include "findlevel.h"
#include "basiccl.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
//#include <iostream>

double sptrsv_syncfree_opencl (const int           *csrColIdx,
                            const int           *csrRowPtr,
                            const VALUE_TYPE    *csrVal,
                                  VALUE_TYPE    *results,
                            const VALUE_TYPE    *b,
                            const int           *levelItem,
                            const int           *levelPtr,
                            double              *leveltime_of_opencl,
                            const int           *nnz_rowperlevel,
                            const int            m,
                            const int            n,
                            const int            nnzTR,
                            const int            device_id,
                            const int            nlevel,
                                    VALUE_TYPE  *svm_results_host);
                            /*const int            substitution,
                            const int            rhs,
                            const int            opt,
                                  VALUE_TYPE    *x,
                            const VALUE_TYPE    *b,
                            const VALUE_TYPE    *x_ref,
                                  double        *gflops*/

int main(int argc, char * argv[]) {
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

    int err = 0;
    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    ocl_command_queue;      // OpenCL Gpu command queues

    
    // platform
    err = basicCL.getNumPlatform(&numPlatforms);            
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    printf("platform number: %i.\n", numPlatforms);

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[device_id], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

            printf("Platform [%i] Vendor: %s Version: %s\n", i, platformVendor, platformVersion);
            printf("Using GPU device: %s ( %i CUs, %lu kB local, %lu MB global, %s )\n",
                   gpuDeviceName, gpuDeviceComputeUnits,
                   gpuDeviceLocalMem / 1024, gpuDeviceGlobalMem / (1024 * 1024), gpuDeviceVersion);

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Gpu commandqueue
    if (1)
        err = basicCL.getCommandQueueProfilingEnable(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    else
        err = basicCL.getCommandQueue(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    //opencl svm
    //int *svm_csr_row_ptr = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+1) * sizeof(int),0);
    //int *svm_csr_col_idx = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+nnzA) * sizeof(int),0);
    //VALUE_TYPE *svm_csr_val = (VALUE_TYPE *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+nnzA) * sizeof(VALUE_TYPE),0);
    VALUE_TYPE *svm_results = (VALUE_TYPE *)clSVMAlloc(cxGpuContext,CL_MEM_READ_WRITE,(m+1) * sizeof(VALUE_TYPE),0);
    //VALUE_TYPE *svm_b = (VALUE_TYPE *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+1) * sizeof(VALUE_TYPE),0);
    //int *svm_level_item = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+1) * sizeof(int),0);
    //int *svm_level_ptr = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(m+1) * sizeof(int),0);
    //int *svm_leveltime_of_opencl = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_WRITE,(nlevel+1) * sizeof(int),0);
    //int *svm_nnz_rowperlevel = (int *)clSVMAlloc(cxGpuContext,CL_MEM_READ_ONLY,(nlevel+1) * sizeof(int),0);


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
    int  *levelItem = (int *)malloc((m+1) * sizeof(int));
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

    double *leveltime_of_omp = (double *)malloc((nlevel+1) * sizeof(double));
    double *leveltime_of_opencl = (double *)malloc((nlevel+1) * sizeof(double));
    memset(leveltime_of_omp, 0, (nlevel+1)  * sizeof(double));
    memset(leveltime_of_opencl, 0, (nlevel+1)  * sizeof(double));

    int counter = 0;//to prevent the same elem
    double s = 0;//temperately memory the past sum

    VALUE_TYPE *b = (double *)malloc(m * sizeof(double));//AX=b
    //printf("Input the bs:\n");
    srand(time(NULL));
    for (int i = 0; i < m; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++)
        {
            b[i] += csrVal[j] * 1;//x[csrcolidx[j]]
        }
        
    }
    /*
    for (int i = 0 ; i < m ; i++) {
        b[i] = 1;//rand()%1800;
        //scanf("%lf",&b[i]); getchar();
    }
    */
    clock_t start,finish; double TheTimes;
    start = clock();//start to clock
    double span;
    struct timeval tvs,tve;
    gettimeofday(&tvs,NULL);
    struct timeval time_begin, time_end;
    for(int loop = 0; loop < BENCH_REPEAT; loop++)
    {//printf("%d\n",loop);

        for (int k = 0; k < nlevel; k++) {//the kth level
        //printf("%d %d\n",levelPtr[k],levelPtr[k+1]);

        gettimeofday(&time_begin,NULL);
    #pragma omp parallel for
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
            //#pragma omp barrier
        gettimeofday(&time_end,NULL);
        leveltime_of_omp[k] += ((time_end.tv_sec-time_begin.tv_sec + (time_end.tv_usec-time_begin.tv_usec)/1000000.0)*1000);

        /*    for (int i = 0; i < m; i++)
                {
                    //printf("%f\n",results[i]);
                }
        */

        }
    }
    gettimeofday(&tve,NULL);
    span = tve.tv_sec-tvs.tv_sec + (tve.tv_usec-tvs.tv_usec)/1000000.0;
    finish = clock();//end the clock
    TheTimes = (double)(finish-start)/CLOCKS_PER_SEC;
    //printf("%f seconds。\n",TheTimes/100);

    for (int i = 0; i < nlevel; i++)
    {
        leveltime_of_omp[i] /= BENCH_REPEAT;
    }
    
    int judge = 1;
    for (int i = 0; i < m; i++)
    {
        if (results[i] != 1)
        {
            judge = 0;
            printf("the wrong result is %d : %f .\n",i,results[i]);
        }
    }
    if (judge)
    {
        printf("THE CPU RESULT IS CORRECT!\n");
    }

    printf("time : %f ms.\n",(span/BENCH_REPEAT)*1000);
    /*
    for (int i = 0; i < 20; i++)
    {
        printf("%1.f ",results[i]);
    }
                printf("\n");
    */

    int *level_rownumber = (int*)malloc((nlevel+1)*sizeof(int));
    for (int i = 0; i < nlevel; i++)
    {
        level_rownumber[i] = levelPtr[i+1] - levelPtr[i];
    }
    int *row_nnz = (int*)malloc((m+1)*sizeof(int));
    for (int i = 0; i < m; i++)
    {
        row_nnz[i] = csrRowPtr[i+1] - csrRowPtr[i];
    }
    int count_nnz = 0;
    int *level_nnz = (int*)malloc((nlevel+1)*sizeof(int));
    memset(level_nnz, 0, nlevel  * sizeof(int));
    for (int i = 0; i < nlevel; i++)
    {
        for (int j = 0; j < level_rownumber[i]; j++)
        {
            level_nnz[i] += row_nnz[levelItem[count_nnz]];
            count_nnz++;
        }
        
    }
    int *nnz_rowperlevel = (int*)malloc((nlevel+1)*sizeof(int));
    for (int i = 0; i < nlevel; i++)
    {
        nnz_rowperlevel[i] = level_nnz[i] / level_rownumber[i];
    }
    /*
    for (int i = 0; i < nlevel; i++)
    {
        printf("rownumber:%d,nnz:%d,per:%d\n",level_rownumber[i],level_nnz[i],nnz_rowperlevel[i]);
    }
    for (int i = 0; i < m; i++)
    {
        printf("rownnz %d\n",row_nnz[i]);
    }*/
    double cltime = sptrsv_syncfree_opencl (csrColIdx,
                            csrRowPtr,
                            csrVal,
                            results,
                            b,
                            levelItem,
                            levelPtr,
                            leveltime_of_opencl,
                            nnz_rowperlevel,
                            m,
                            m,
                            nnzTR,
                            device_id,
                            nlevel,
                            svm_results);
                            /*substitution,
                            rhs,
                            opt,
                            *x,
                            *b,
                            *x_ref,
                            *gflops*/

    double oracle = 0;
    for (int i = 0; i < nlevel; i++)
    {
        //printf("the %d level has %d items, time of omp is %f, time of opencl is %f.\n",i,levelPtr[i+1]-levelPtr[i],leveltime_of_omp[i],leveltime_of_opencl[i]);
        if (leveltime_of_omp[i] < leveltime_of_opencl[i])
        {
            printf("%d,%d,%f,%f,%f\n",i,levelPtr[i+1]-levelPtr[i],leveltime_of_omp[i],leveltime_of_opencl[i],leveltime_of_omp[i]);
            oracle += leveltime_of_omp[i];
        }else
        {
            printf("%d,%d,%f,%f,%f\n",i,levelPtr[i+1]-levelPtr[i],leveltime_of_omp[i],leveltime_of_opencl[i],leveltime_of_opencl[i]);
            oracle += leveltime_of_opencl[i];
        }
    }
    printf("finally %f %f %f\n",(span/BENCH_REPEAT)*1000,cltime,oracle);
    
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
    free(leveltime_of_omp);
    free(leveltime_of_opencl);
    free(level_rownumber);
    free(row_nnz);
    free(level_nnz);
    free(nnz_rowperlevel);

    //clSVMFree(cxGpuContext,svm_csr_row_ptr);
    //clSVMFree(cxGpuContext,svm_csr_col_idx);
    //clSVMFree(cxGpuContext,svm_csr_val);
    clSVMFree(cxGpuContext,svm_results);
    //clSVMFree(cxGpuContext,svm_b);
    //clSVMFree(cxGpuContext,svm_level_item);
    //clSVMFree(cxGpuContext,svm_level_ptr);
    //clSVMFree(cxGpuContext,svm_leveltime_of_opencl);
    //clSVMFree(cxGpuContext,svm_nnz_rowperlevel);

    return 0;
}


double sptrsv_syncfree_opencl ( const int           *csrColIdx,
                                const int           *csrRowPtr,
                                const VALUE_TYPE    *csrVal,
                                      VALUE_TYPE    *results,
                                const VALUE_TYPE    *b,
                                const int           *levelItem,
                                const int           *levelPtr,
                                double              *leveltime_of_opencl,
                                const int           *nnz_rowperlevel,
                                const int            m,
                                const int            n,
                                const int            nnzTR,
                                const int            device_id,
                                const int            nlevel,
                                        VALUE_TYPE  *svm_results_host)
                                /*const int            substitution,
                                const int            rhs,
                                const int            opt,
                                VALUE_TYPE           *x,
                                const VALUE_TYPE    *b,
                                const VALUE_TYPE    *x_ref,
                                double              *gflops*/
{
    int rhs = 1;
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    int err = 0;

    // set device
    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    ocl_command_queue;      // OpenCL Gpu command queues

    bool profiling = true;
    
    // platform
    err = basicCL.getNumPlatform(&numPlatforms);            
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    printf("platform number: %i.\n", numPlatforms);

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[device_id], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

            printf("Platform [%i] Vendor: %s Version: %s\n", i, platformVendor, platformVersion);
            printf("Using GPU device: %s ( %i CUs, %lu kB local, %lu MB global, %s )\n",
                   gpuDeviceName, gpuDeviceComputeUnits,
                   gpuDeviceLocalMem / 1024, gpuDeviceGlobalMem / (1024 * 1024), gpuDeviceVersion);

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Gpu commandqueue
    if (1)
        err = basicCL.getCommandQueueProfilingEnable(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    else
        err = basicCL.getCommandQueue(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    const char *ocl_source_code_sptrsv =
    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                                          \n"
    "                                                                                                                           \n"
    "    #ifndef VALUE_TYPE                                                                                                     \n"
    "    #define VALUE_TYPE float                                                                                               \n"
    "    #endif                                                                                                                 \n"
    "    #define WARP_SIZE 64                                                                                                   \n"
    "    #define THREADS_PER_BLOCK 256                                                                                          \n"
    "	inline                                                                                                      	        \n"
    "	void sum_64(__local volatile VALUE_TYPE *s_sum,	                                                                        \n"
    "	const int local_id)	                                                                                                    \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 32];	                                                                            \n"
    "	s_sum[local_id] += s_sum[local_id + 16];	                                                                            \n"
    "	s_sum[local_id] += s_sum[local_id + 8];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 4];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 2];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "	void sum_32(__local volatile VALUE_TYPE *s_sum,	                                                                        \n"
    "	const int local_id)	                                                                                                    \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 16];	                                                                            \n"
    "	s_sum[local_id] += s_sum[local_id + 8];                                                                         	    \n"
    "	s_sum[local_id] += s_sum[local_id + 4];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 2];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "	void sum_16(__local volatile VALUE_TYPE *s_sum,	                                                                        \n"
    "	const int local_id)	                                                                                                    \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 8];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 4];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 2];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "	void sum_8(__local volatile VALUE_TYPE *s_sum,      	                                                                \n"
    "	const int local_id)	                                                                                                    \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 4];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 2];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "	void sum_4(__local volatile VALUE_TYPE *s_sum,	                                                                        \n"
    "	const int local_id)	                                                                                                    \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 2];	                                                                                \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "	void sum_2(__local volatile VALUE_TYPE *s_sum,	                                                                        \n"
    "	const int local_id) 	                                                                                                \n"
    "	{	                                                                                                                    \n"
    "	s_sum[local_id] += s_sum[local_id + 1];	                                                                                \n"
    "	//VALUE_TYPE sum = s_sum[local_id];	                                                                                    \n"
    "	//if (local_id < 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16] + s_sum[local_id + 32] + s_sum[local_id + 48];	\n"
    "	//if (local_id < 4) s_sum[local_id] = sum = sum + s_sum[local_id + 4] + s_sum[local_id + 8] + s_sum[local_id + 12];	    \n"
    "	//if (local_id < 1) s_sum[local_id] = sum = sum + s_sum[local_id + 1] + s_sum[local_id + 2] + s_sum[local_id + 3];	    \n"
    "	}                                                                                                               	    \n"
    "    __kernel                                                                                                               \n"
    "    void sptrsv_syncfree_opencl_executor(__global const int            *d_csrColIdx,                                       \n"
    "                                         __global const int            *d_csrRowPtr,                                       \n"
    "                                         __global const VALUE_TYPE     *d_csrVal,                                          \n"
    "                                         __global VALUE_TYPE           *d_results,                                         \n"
    "                                         __global VALUE_TYPE           *d_b,                                               \n"
    "                                         __global const int            *d_levelItem,                                       \n"
    "                                         const int                      levelstart,                                        \n"
    "                                         const int                      levelend,                                          \n"
    "                                         const int ROWS_PER_BLOCK,                                                         \n"
    "                                         const int THREADS_PER_ROW,                                                        \n"
    "                                         volatile __local VALUE_TYPE   *s_sum,                                             \n"
    "                                         __global VALUE_TYPE           *svm_results)                                       \n"
    "    {                                                                                                                      \n"
    "       svm_results[1] = 19613998;                                                                                          \n"
    "       int global_id = get_global_id(0);                                                                                   \n"
    "       int local_id = get_local_id(0);                                                                                     \n"
    "       int thread_lane = local_id % THREADS_PER_ROW;                                                                       \n"
    "       //int level_id = get_global_id(0);                                                                                  \n"
    "       int row_lane = local_id / THREADS_PER_ROW;                                                                          \n"
    "       int num_rows = ROWS_PER_BLOCK * get_num_groups(0);                                                                  \n"
    "       const int row_item   = get_global_id(0)   /  THREADS_PER_ROW;                                                       \n"
    "	    //const int row_item = get_group_id(0);	                                                                            \n"
    "	    const int local_size = get_local_size(0);	                                                                        \n"
    "                                                                                                                           \n"
    "       for(int row = row_item; row < levelend - levelstart; row += num_rows)                                               \n"
    "		{                                                                                                                   \n"
    "	        //volatile __local VALUE_TYPE s_sum[THREADS_PER_BLOCK]; 	                                                    \n"
    "           //if (row_item+levelstart < levelend){                                                                          \n"
    "	        int csrRowId = d_levelItem[levelstart+row];                                                         	        \n"
    "           //barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
    "           int row_start =  d_csrRowPtr[csrRowId];                                                                         \n"
    "           int row_end =  d_csrRowPtr[csrRowId+1]-1;                                                                       \n"
    "	        VALUE_TYPE sum = 0;                                                                                         	\n"
    "	        //begin to solve the each thread                                                            	                \n"
    "           //int i = d_csrRowPtr[csrRowId]+local_id;                                                                       \n"
    "           //while (i < d_csrRowPtr[csrRowId+1]-1)                                                                         \n"
    "           //{                                                                                                             \n"
    "           //    sum += (d_csrVal[i] * d_results[d_csrColIdx[i]]);                                                         \n"
    "           //    i+=local_size;                                                                                            \n"
    "           //}                                                                                                             \n"
    "           if (THREADS_PER_ROW == 64 && row_end - row_start > 64)                                                          \n"
    "           {                                                                                                               \n"
    "               // ensure aligned memory access to d_csrColIdx and d_csrVal                                                 \n"
    "                                                                                                                           \n"
    "               int jj = row_start - (row_start & (THREADS_PER_ROW - 1)) + thread_lane;                                     \n"
    "                                                                                                                           \n"
    "               // accumulate local sums                                                                                    \n"
    "               if(jj >= row_start && jj < row_end)                                                                         \n"
    "               sum += d_csrVal[jj] * d_results[d_csrColIdx[jj]];                                                           \n"
    "                                                                                                                           \n"
    "               // accumulate local sums                                                                                    \n"
    "               for(jj += THREADS_PER_ROW; jj < row_end; jj += THREADS_PER_ROW)                                             \n"
    "                   sum += d_csrVal[jj] * d_results[d_csrColIdx[jj]];                                                       \n"
    "           }                                                                                                               \n"
    "           else                                                                                                            \n"
    "           {                                                                                                               \n"
    "               // accumulate local sums                                                                                    \n"
    "               for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_ROW)                                  \n"
    "                   sum += d_csrVal[jj] * d_results[d_csrColIdx[jj]];                                                       \n"
    "           }                                                                                                               \n"
    "                                                                                                                           \n"
    "	        //for (int i = d_csrRowPtr[csrRowId]+thread_lane; i < d_csrRowPtr[csrRowId+1]-1; i+=THREADS_PER_ROW)	        \n"
    "	        //{//solve the last one	                                                                                        \n"
    "	        //    sum += (d_csrVal[i] * d_results[d_csrColIdx[i]]);	                                                        \n"
    "	        //}	                                                                                                            \n"
    "           //barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
    "	        s_sum[local_id] = sum; 	                                                                                        \n"
    "           //barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
    "           if (THREADS_PER_ROW > 32) s_sum[local_id] = sum = sum + s_sum[local_id + 32];                                   \n"
    "           if (THREADS_PER_ROW > 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16];                                   \n"
    "           if (THREADS_PER_ROW >  8) s_sum[local_id] = sum = sum + s_sum[local_id +  8];                                   \n"
    "           if (THREADS_PER_ROW >  4) s_sum[local_id] = sum = sum + s_sum[local_id +  4];                                   \n"
    "           if (THREADS_PER_ROW >  2) s_sum[local_id] = sum = sum + s_sum[local_id +  2];                                   \n"
    "           if (THREADS_PER_ROW >  1) s_sum[local_id] = sum = sum + s_sum[local_id +  1];                                   \n"
    "           //if (THREADS_PER_ROW > 32) {sum_64(s_sum, thread_lane);}                                                       \n"
    "           //else if (THREADS_PER_ROW > 16) {sum_32(s_sum, thread_lane);}                                                  \n" 
    "           //else if (THREADS_PER_ROW > 8) {sum_16(s_sum, thread_lane);}                                                   \n"
    "           //else if (THREADS_PER_ROW > 4) {sum_8(s_sum, thread_lane);}                                                    \n"
    "           //else if (THREADS_PER_ROW > 2) {sum_4(s_sum, thread_lane);}                                                    \n" 
    "           //else if (THREADS_PER_ROW > 1) {sum_2(s_sum, thread_lane);}                                                    \n"
    "	        //sum_64(s_sum, thread_lane);	                                                                                \n"
    "           //barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
    "	        //sum = s_sum[thread_lane]; 	                                                                                \n"
    "           //barrier(CLK_LOCAL_MEM_FENCE);                                                                                 \n"
    "	        if (!thread_lane) 	                                                                                            \n"
    "	        {                                                   	                                                        \n"
    "                                                                                                                           \n"
    "               //VALUE_TYPE OOOO = d_b[csrRowId] - s_sum[local_id]; //each thread is finished                              \n"
    "               d_results[csrRowId] = (d_b[csrRowId] - s_sum[local_id]) / d_csrVal[d_csrRowPtr[csrRowId+1]-1];              \n"
    "               //barrier(CLK_LOCAL_MEM_FENCE);                                                                             \n"
    "               //d_results[1] = csrRowId;                                                                                  \n"
    "           }                                                                                                               \n"
    "       }                                                                                                                   \n"
    "   }                                                                                                                       \n";

    // Create the program
    cl_program          ocl_program_sptrsv;

    size_t source_size_sptrsv[] = { strlen(ocl_source_code_sptrsv)};

    ocl_program_sptrsv = clCreateProgramWithSource(cxGpuContext, 1, &ocl_source_code_sptrsv, source_size_sptrsv, &err);

    if(err != CL_SUCCESS) {printf("OpenCL clCreateProgramWithSource ERROR CODE = %i\n", err); return err;}

    // Build the program

    if (sizeof(VALUE_TYPE) == 8)
        err = clBuildProgram(ocl_program_sptrsv, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=double", NULL, NULL);
    else
        err = clBuildProgram(ocl_program_sptrsv, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=float", NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL clBuildProgram ERROR CODE = %i\n", err); return err;}
    
    // Create kernels
    cl_kernel  ocl_kernel_sptrsv_levelset;
    ocl_kernel_sptrsv_levelset = clCreateKernel(ocl_program_sptrsv, "sptrsv_syncfree_opencl_executor", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}

    // transfer host mem to device mem
    // Define pointers of matrix L, vector x and b
    cl_mem      d_csrColIdx;
    cl_mem      d_csrRowPtr;
    cl_mem      d_csrVal;
    cl_mem      d_b;
    cl_mem      d_results;
    cl_mem      d_levelItem;

    //cl_mem      svm_csrColIdx;
    //cl_mem      svm_csrRowPtr;
    //cl_mem      svm_csrVal;
    //cl_mem      svm_b;
    cl_mem      svm_results;
    //cl_mem      svm_levelItem;

    /*svm_csrColIdx = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    svm_csrRowPtr = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (n+1)  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    svm_csrVal    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, svm_csrColIdx, CL_TRUE, 0, nnzTR * sizeof(int), csrColIdx, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, svm_csrRowPtr, CL_TRUE, 0, (n+1)  * sizeof(int), csrRowPtr, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, svm_csrVal, CL_TRUE, 0, nnzTR  * sizeof(VALUE_TYPE), csrVal, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    svm_b    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, m * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, svm_b, CL_TRUE, 0, m * rhs * sizeof(VALUE_TYPE), b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}*/
    svm_results    = clCreateBuffer(cxGpuContext, 
                                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                    n * rhs * sizeof(VALUE_TYPE),
                                    svm_results_host,
                                    &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    //memset(results, 0, m  * sizeof(VALUE_TYPE));
    /*err = clEnqueueWriteBuffer(ocl_command_queue, svm_results, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), results, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    svm_levelItem    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n * rhs * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, svm_levelItem, CL_TRUE, 0, n * sizeof(int), levelItem, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}*/



    // Matrix L
    d_csrColIdx = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_csrRowPtr = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (n+1)  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_csrVal    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrColIdx, CL_TRUE, 0, nnzTR * sizeof(int), csrColIdx, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowPtr, CL_TRUE, 0, (n+1)  * sizeof(int), csrRowPtr, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrVal, CL_TRUE, 0, nnzTR  * sizeof(VALUE_TYPE), csrVal, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector b
    d_b    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, m * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_b, CL_TRUE, 0, m * rhs * sizeof(VALUE_TYPE), b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector x
    d_results    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    memset(results, 0, m  * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_results, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), results, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // level
    d_levelItem    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n * rhs * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_levelItem, CL_TRUE, 0, n * sizeof(int), levelItem, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    unsigned long szLocalWorkSize[1];
    unsigned long szGlobalWorkSize[1];
    const int THREADS_PER_BLOCK  = 256;
    //int num_threads = 1 * WARP_SIZE; 
    //szLocalWorkSize[0]  = num_threads;
    int levelstart;
    int levelend;

    err  = clSetKernelArg(ocl_kernel_sptrsv_levelset, 0,  sizeof(cl_mem), (void*)&d_csrColIdx);
    err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 1,  sizeof(cl_mem), (void*)&d_csrRowPtr);
    err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 2,  sizeof(cl_mem), (void*)&d_csrVal);
    err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 3,  sizeof(cl_mem), (void*)&d_results);
    err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 4,  sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 5,  sizeof(cl_mem), (void*)&d_levelItem);

    VALUE_TYPE *results_tmp = (VALUE_TYPE *)malloc(m * sizeof(VALUE_TYPE));//the results
    if (results_tmp == NULL)
    {
        printf("NULL\n");
    }

    double time_opencl_analysis = 0;
    for(int loop = 0; loop < BENCH_REPEAT; loop++)
    {//printf("%d\n",loop);
        for (int k = 0; k < nlevel; k++) {//the kth level
            int THREADS_PER_ROW;
            levelstart = levelPtr[k];
            levelend = levelPtr[k+1];
            if (nnz_rowperlevel[k] <=  2) {
                THREADS_PER_ROW = 2;
            }
            else if (nnz_rowperlevel[k] <=  4) {
                THREADS_PER_ROW = 4;
            }
            else if (nnz_rowperlevel[k] <=  8) {
                THREADS_PER_ROW = 8;
            }
            else if (nnz_rowperlevel[k] <= 16) {
                THREADS_PER_ROW = 16;
            }
            else if (nnz_rowperlevel[k] <= 32) {
                THREADS_PER_ROW = 32;
            }
            else
                THREADS_PER_ROW = 64;
            int num_threads = THREADS_PER_BLOCK;
            szLocalWorkSize[0]  = num_threads;
            int num_blocks = ceil ((double)(levelend-levelstart) / (double)(num_threads/THREADS_PER_ROW));
            szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];
            int ROWS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_ROW;
            //printf("%d %d\n",levelstart,levelend);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 6,  sizeof(cl_int), (void*)&levelstart);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 7,  sizeof(cl_int), (void*)&levelend);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 8,  sizeof(cl_int), (void*)&ROWS_PER_BLOCK);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 9,  sizeof(cl_int), (void*)&THREADS_PER_ROW);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 10, sizeof(VALUE_TYPE) * (ROWS_PER_BLOCK * THREADS_PER_ROW + THREADS_PER_ROW / 2), NULL);
            err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 11,  sizeof(cl_mem), (void*)&svm_results);

            //int num_blocks = ceil ((double)(levelend-levelstart) / (double)(num_threads/WARP_SIZE));
            err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsv_levelset, 1,
                                                NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
                    if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsv_levelset kernel run error = %i\n", err); return err; }

            err = clWaitForEvents(1, &ceTimer);
            if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }
            basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
            time_opencl_analysis += double(endTime - startTime) / 1000000.0;
            leveltime_of_opencl[k] += double(endTime - startTime) / 1000000.0;
            //printf("opencl SpTRSV used %4.6f ms   the level is %d items: %d \n", time_opencl_analysis,k,levelend-levelstart);
            //err = clEnqueueReadBuffer(ocl_command_queue, d_results, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), results, 0, NULL, NULL);
            //if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
            /*for (int i = 0; i < m; i++)
            {
                //printf("%f\n",results[i]);
            }*/
        }

        if (loop==0)
        {
            err = clEnqueueReadBuffer(ocl_command_queue, d_results, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), results, 0, NULL, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
            for (int i = 0; i < m; i++)
            {
                results_tmp[i] = results[i];
            }
        }
    }

    for (int i = 0; i < nlevel; i++)
    {
        leveltime_of_opencl[i] /= BENCH_REPEAT;
    }
    printf("opencl SpTRSV used %4.6f ms\n", time_opencl_analysis/BENCH_REPEAT);
        
    int judge = 0;
    for (int i = 0; i < m; i++)
    {
        if (results_tmp[i] != 1)
        {
            judge++;
            //printf("the wrong result is %d : %f.\n",i,results_tmp[i]);
        }
    }
    if (!judge)
    {
        printf("THE GPU RESULT IS CORRECT!\n");
    }else
    {
        printf("the number of wrong results : %d\n",judge);
    }

    /*
    for (int i = 0; i < m; i++)
    {
        printf("%f\n",results_tmp[i]);
    }
    */
    //  free resources
    free(results_tmp);
    if(d_csrColIdx) err = clReleaseMemObject(d_csrColIdx); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrRowPtr) err = clReleaseMemObject(d_csrRowPtr); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrVal)    err = clReleaseMemObject(d_csrVal); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_b) err = clReleaseMemObject(d_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_results) err = clReleaseMemObject(d_results); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_levelItem) err = clReleaseMemObject(d_levelItem); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    //if(svm_csrColIdx) err = clReleaseMemObject(svm_csrColIdx); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    //if(svm_csrRowPtr) err = clReleaseMemObject(svm_csrRowPtr); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    //if(svm_csrVal)    err = clReleaseMemObject(svm_csrVal); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    //if(svm_b) err = clReleaseMemObject(svm_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(svm_results) err = clReleaseMemObject(svm_results); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    //if(svm_levelItem) err = clReleaseMemObject(svm_levelItem); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    return time_opencl_analysis/BENCH_REPEAT;

}
