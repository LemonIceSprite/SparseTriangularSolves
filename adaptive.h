#ifndef ADAPTIVE_H
#define ADAPTIVE_H
#include "common.h"

//  0:use cpu
//  1:use gpu

int selction (int nnz_rowperlevel){
    if (nnz_rowperlevel > THRESHOLD_VALUE)
    {
        return 1;
    }else     return 0;
    
}

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
                               VALUE_TYPE  *svm_results_host)
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
    "    #ifndef VALUE_TYPE                                                                                                     \n"
    "    #define VALUE_TYPE float                                                                                               \n"
    "    #endif                                                                                                                 \n"
    "    #define WARP_SIZE 64                                                                                                   \n"
    "    #define THREADS_PER_BLOCK 256                                                                                          \n"
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
    "                                                                                                                           \n"
    "       int global_id = get_global_id(0);                                                                                   \n"
    "       int local_id = get_local_id(0);                                                                                     \n"
    "       int thread_lane = local_id % THREADS_PER_ROW;                                                                       \n"
    "       int row_lane = local_id / THREADS_PER_ROW;                                                                          \n"
    "       int num_rows = ROWS_PER_BLOCK * get_num_groups(0);                                                                  \n"
    "       const int row_item   = get_global_id(0)   /  THREADS_PER_ROW;                                                       \n"
    "	    const int local_size = get_local_size(0);	                                                                        \n"
    "                                                                                                                           \n"
    "       for(int row = row_item; row < levelend - levelstart; row += num_rows)                                               \n"
    "		{                                                                                                                   \n"
    "	        int csrRowId = d_levelItem[levelstart+row];                                                         	        \n"
    "           int row_start =  d_csrRowPtr[csrRowId];                                                                         \n"
    "           int row_end =  d_csrRowPtr[csrRowId+1]-1;                                                                       \n"
    "	        VALUE_TYPE sum = 0;                                                                                         	\n"
    "           if (THREADS_PER_ROW == 64 && row_end - row_start > 64)                                                          \n"
    "           {                                                                                                               \n"
    "               // ensure aligned memory access to d_csrColIdx and d_csrVal                                                 \n"
    "               int jj = row_start - (row_start & (THREADS_PER_ROW - 1)) + thread_lane;                                     \n"
    "               // accumulate local sums                                                                                    \n"
    "               if(jj >= row_start && jj < row_end)                                                                         \n"
    "               sum += d_csrVal[jj] * svm_results[d_csrColIdx[jj]];                                                         \n"
    "                                                                                                                           \n"
    "               // accumulate local sums                                                                                    \n"
    "               for(jj += THREADS_PER_ROW; jj < row_end; jj += THREADS_PER_ROW)                                             \n"
    "                   sum += d_csrVal[jj] * svm_results[d_csrColIdx[jj]];                                                     \n"
    "           }                                                                                                               \n"
    "           else                                                                                                            \n"
    "           {                                                                                                               \n"
    "               // accumulate local sums                                                                                    \n"
    "               for(int jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_ROW)                                  \n"
    "                   sum += d_csrVal[jj] * svm_results[d_csrColIdx[jj]];                                                     \n"
    "           }                                                                                                               \n"
    "                                                                                                                           \n"
    "	        s_sum[local_id] = sum; 	                                                                                        \n"
    "           if (THREADS_PER_ROW > 32) s_sum[local_id] = sum = sum + s_sum[local_id + 32];                                   \n"
    "           if (THREADS_PER_ROW > 16) s_sum[local_id] = sum = sum + s_sum[local_id + 16];                                   \n"
    "           if (THREADS_PER_ROW >  8) s_sum[local_id] = sum = sum + s_sum[local_id +  8];                                   \n"
    "           if (THREADS_PER_ROW >  4) s_sum[local_id] = sum = sum + s_sum[local_id +  4];                                   \n"
    "           if (THREADS_PER_ROW >  2) s_sum[local_id] = sum = sum + s_sum[local_id +  2];                                   \n"
    "           if (THREADS_PER_ROW >  1) s_sum[local_id] = sum = sum + s_sum[local_id +  1];                                   \n"
    "	        if (!thread_lane) 	                                                                                            \n"
    "	        {                                                   	                                                        \n"
    "               svm_results[csrRowId] = (d_b[csrRowId] - s_sum[local_id]) / d_csrVal[d_csrRowPtr[csrRowId+1]-1];            \n"
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
    
    cl_mem      svm_results;
    svm_results    = clCreateBuffer(cxGpuContext,
                                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                    n * rhs * sizeof(VALUE_TYPE),
                                    svm_results_host,
                                    &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    
    
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
    
    struct timeval time_begin, time_end;
    double time_opencl_analysis = 0;
    double *leveltime_of_omp = (double *)malloc((nlevel+1)*sizeof(double));
    memset(leveltime_of_omp, 0, (nlevel+1)  * sizeof(double));
    
    //==================================================================    pure opencl   ========================================================
    double time_opencl_analysis_tmp = 0;
    double *leveltime_of_opencl_tmp = (double *)malloc((nlevel+1) * sizeof(double));
    memset(leveltime_of_opencl_tmp, 0, (nlevel+1)  * sizeof(double));
    for(int loop = 0; loop < BENCH_REPEAT; loop++)
    {//printf("%d\n",loop);
        for (int k = 0; k < nlevel; k++) {//the kth level
            //if (selction(levelPtr[k+1] - levelPtr[k]))
            if(1)
                //if(0)
            {
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
                time_opencl_analysis_tmp += double(endTime - startTime) / 1000000.0;
                //printf("%f\n",time_opencl_analysis);
                leveltime_of_opencl_tmp[k] += double(endTime - startTime) / 1000000.0;
            }
            else
            { }
        }
        
    }
    //==================================================================    pure opencl   ========================================================
    
    struct timeval time_begin_tmp, time_end_tmp;
    double *leveltime_of_omp_tmp = (double *)malloc((nlevel+1)*sizeof(double));
    memset(leveltime_of_omp_tmp, 0, (nlevel+1)  * sizeof(double));
    //==================================================================    pure openmp   ========================================================
    for(int loop = 0; loop < BENCH_REPEAT; loop++)
    {//printf("%d\n",loop);
        for (int k = 0; k < nlevel; k++) {//the kth level
            //if (selction(levelPtr[k+1] - levelPtr[k]))
            //if(1)
            if(0)
            {}
            else
            {
                gettimeofday(&time_begin_tmp,NULL);
#pragma omp parallel for
                for (int j = levelPtr[k] ; j < levelPtr[k+1]; j++) {//parallel the level k
                    int i = levelItem[j];//the row need be solved
                    int s = 0;
                    int now = csrRowPtr[i];
                    while(now < csrRowPtr[i+1]-1){
                        s += (csrVal[now] * svm_results_host[csrColIdx[now]]);
                        now++;
                    }
                    svm_results_host[i] = (b[i] - s) / csrVal[now];
                }
                //#pragma omp barrier
                gettimeofday(&time_end_tmp,NULL);
                leveltime_of_omp_tmp[k] += ((time_end_tmp.tv_sec-time_begin_tmp.tv_sec + (time_end_tmp.tv_usec-time_begin_tmp.tv_usec)/1000000.0)*1000);
            }
        }
        
    }
    //==================================================================    pure openmp   ========================================================
    
    //==================================================================      adaptive    ========================================================
    
    for(int loop = 0; loop < BENCH_REPEAT; loop++)
    {//printf("%d\n",loop);
        for (int k = 0; k < nlevel; k++) {//the kth level
            if (selction(levelPtr[k+1] - levelPtr[k]))
                //if(1)
                //if(0)
            {
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
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 6,  sizeof(cl_int), (void*)&levelstart);
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 7,  sizeof(cl_int), (void*)&levelend);
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 8,  sizeof(cl_int), (void*)&ROWS_PER_BLOCK);
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 9,  sizeof(cl_int), (void*)&THREADS_PER_ROW);
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 10, sizeof(VALUE_TYPE) * (ROWS_PER_BLOCK * THREADS_PER_ROW + THREADS_PER_ROW / 2), NULL);
                err |= clSetKernelArg(ocl_kernel_sptrsv_levelset, 11,  sizeof(cl_mem), (void*)&svm_results);
                err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsv_levelset, 1,
                                             NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
                if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsv_levelset kernel run error = %i\n", err); return err; }
                err = clWaitForEvents(1, &ceTimer);
                if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }
                basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
                time_opencl_analysis += double(endTime - startTime) / 1000000.0;
                leveltime_of_opencl[k] += double(endTime - startTime) / 1000000.0;
            }
            else
            {
                gettimeofday(&time_begin,NULL);
#pragma omp parallel for
                for (int j = levelPtr[k] ; j < levelPtr[k+1]; j++) {//parallel the level k
                    int i = levelItem[j];//the row need be solved
                    int s = 0;
                    int now = csrRowPtr[i];
                    while(now < csrRowPtr[i+1]-1){
                        s += (csrVal[now] * svm_results_host[csrColIdx[now]]);
                        now++;
                    }
                    svm_results_host[i] = (b[i] - s) / csrVal[now];
                }
                //#pragma omp barrier
                gettimeofday(&time_end,NULL);
                leveltime_of_omp[k] += ((time_end.tv_sec-time_begin.tv_sec + (time_end.tv_usec-time_begin.tv_usec)/1000000.0)*1000);
            }
        }
    }
    //==================================================================      adaptive    ========================================================
    
    double time_openmp_analysis = 0;
    double time_openmp_analysis_tmp = 0;
    
    for (int i = 0; i < nlevel; i++)
    {
        leveltime_of_opencl[i] /= BENCH_REPEAT;
        leveltime_of_omp[i] /= BENCH_REPEAT;
        leveltime_of_opencl_tmp[i] /= BENCH_REPEAT;
        leveltime_of_omp_tmp[i] /= BENCH_REPEAT;
        time_openmp_analysis += leveltime_of_omp[i];
        time_openmp_analysis_tmp += leveltime_of_omp_tmp[i];
    }
    printf("opencl SpTRSV used %4.6f ms\n", time_opencl_analysis/BENCH_REPEAT);
    printf("openmp SpTRSV used %4.6f ms\n", time_openmp_analysis);
    printf("adaptive used %4.6f ms\n", time_openmp_analysis + time_opencl_analysis/BENCH_REPEAT);
    printf("pure opencl SpTRSV used %4.6f ms\n", time_opencl_analysis_tmp/BENCH_REPEAT);
    printf("pure openmp SpTRSV used %4.6f ms\n", time_openmp_analysis_tmp);
    
    int judge = 0;
    for (int i = 0; i < m; i++)
    {
        if (svm_results_host[i] != 1)
        {
            judge++;
            //printf("the wrong result is %d : %f.\n",i,results_tmp[i]);
        }
    }
    if (!judge)
    {
        printf("THE RESULT IS CORRECT!\n");
    }else
    {
        printf("the number of wrong results : %d\n",judge);
    }
    
    for (int i = 0; i < nlevel; i++)
    {
        //printf("%f,%f,%f,%f\n",leveltime_of_omp[i],leveltime_of_opencl[i],leveltime_of_omp_tmp[i] / BENCH_REPEAT,leveltime_of_opencl_tmp[i] / BENCH_REPEAT);
    }
    
    //  free resources
    free(results_tmp);
    free(leveltime_of_omp);
    free(leveltime_of_opencl_tmp);
    free(leveltime_of_omp_tmp);
    if(d_csrColIdx) err = clReleaseMemObject(d_csrColIdx); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrRowPtr) err = clReleaseMemObject(d_csrRowPtr); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrVal)    err = clReleaseMemObject(d_csrVal); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_b) err = clReleaseMemObject(d_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_results) err = clReleaseMemObject(d_results); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_levelItem) err = clReleaseMemObject(d_levelItem); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    
    if(svm_results) err = clReleaseMemObject(svm_results); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    
    return time_opencl_analysis/BENCH_REPEAT;
    
}

#endif
