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
#include "adaptive.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
//#include <iostream>

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

    //opencl svm
    VALUE_TYPE *svm_results = (VALUE_TYPE *)clSVMAlloc(cxGpuContext,CL_MEM_READ_WRITE,(m+1) * sizeof(VALUE_TYPE),0);

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
    for (int i = 0; i < m; i++)
    {
        b[i] = 0;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++)
        {
            b[i] += csrVal[j] * 1;//x[csrcolidx[j]]
        }
        
    }
    
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
  
    free(levelItem);
    free(levelPtr);
    free(csrColIdx);
    free(csrVal);
    free(csrRowPtr);
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowPtr_tmp);
    free(results);
    free(b);
    free(leveltime_of_omp);
    free(leveltime_of_opencl);
    free(level_rownumber);
    free(row_nnz);
    free(level_nnz);
    free(nnz_rowperlevel);
    
    clSVMFree(cxGpuContext,svm_results);

    return 0;
}
