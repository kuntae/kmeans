#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <CL/cl.h>

#define error(err) if(err !=CL_SUCCESS){fprintf(stderr, "[%s:%d] error: %d\n",__FILE__, __LINE__, err);exit(EXIT_FAILURE);}
#define NUM_THREAD 4
#define ASSIGN_LWS 256
#define LWS 16
typedef struct Centroid_Data_t{
    struct Point p;
    int cnt;
}Centroid_Data_t;

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    //variable for OpenCL.
    cl_platform_id plat;
    cl_device_id dev;
    cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
    cl_context context;
    cl_command_queue q;
    cl_kernel assign, update_map, update_reduce;
    cl_program pg;
    cl_int err;
    cl_event ev_buf, ev_assign, ev_update;
    size_t assign_lws[1], assign_gws[1];
    size_t update_lws[2], update_gws[2];

    //Nomal variable.
    int data_div_class, map_wg_size;
    size_t data_size, centroid_size, partitioned_size, tmpcent_size;
    //loop variable.
    int i, data_i, class_i;

    data_div_class = data_n / class_n;
    map_wg_size = data_div_class / LWS;

    data_size = sizeof(Point) * data_n;
    centroid_size = sizeof(Point) * class_n;
    partitioned_size = sizeof(int) * data_n;
    tmpcent_size = sizeof(Centroid_Data_t) * data_div_class; 

    //count number of data in each class
    int *count = (int*)malloc(class_n* sizeof(int));
    if(count == NULL){
        fprintf(stderr, "%d line allocate error\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    Centroid_Data_t *tmp_centroid = (Centroid_Data_t*)malloc(tmpcent_size);
    if(tmp_centroid == NULL){
        fprintf(stderr, "%d line allocate error\n", __LINE__);
        exit(EXIT_FAILURE);
    }
    Centroid_Data_t *finally = (Centroid_Data_t*)malloc
        (sizeof(Centroid_Data_t) * map_wg_size / LWS * class_n);
    //==========================================OpenCL========================================
    //Get platform id.
    err = clGetPlatformIDs(1, &plat, NULL);
    error(err);

    //Get device id.
    err = clGetDeviceIDs(plat, dev_type, 1, &dev, NULL);
    error(err);

    //Create context.
    context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    error(err);

    //Create command queue.
    q = clCreateCommandQueue(context, dev, 0, &err);
    error(err);

    //Create a program.
    size_t len;
    char *src;
    FILE *fp = fopen("cp_kmeans_kernel.cl", "r");
    fseek(fp, 0, SEEK_END);
    len = ftell(fp);
    rewind(fp);
    src = (char*)malloc(sizeof(char) * len+1);
    fread(src, sizeof(char), len, fp);
    src[len] = '\0';
    pg = clCreateProgramWithSource(context,1, (const char**)&src, &len, &err);
    free(src);
    len = 0;
    error(err);

    //Build a program.
    char option[100];
    sprintf(option, 
            "-DCLASS_N=%d -DDATA_N=%d -DWG=%d -DASSIGN_LWS=%d -DUPDATE_SIZE=%d",
            class_n, data_n, map_wg_size, ASSIGN_LWS, LWS);
    err = clBuildProgram(pg, 1, &dev, option, NULL, NULL);
    if(err != CL_SUCCESS){
        char *log;
        clGetProgramBuildInfo(pg, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        log = (char*)malloc(sizeof(char) * len +1);
        clGetProgramBuildInfo(pg, dev, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        log[len] = '\0';
        fprintf(stderr, "===================================================");
        fprintf(stderr, "%s\n", log);
        fprintf(stderr, "===================================================");
        free(log);
        exit(EXIT_FAILURE);
    }

    //Create kernel.
    assign = clCreateKernel(pg, "assign", &err);
    error(err);
    update_map = clCreateKernel(pg, "update_map", &err);
    error(err);
    update_reduce = clCreateKernel(pg, "update_reduce", &err);
    error(err);

    //Variable for buffer object.
    cl_mem mCentroids, mPartitioned, mData, mTmpCent, mFinally;

    //Create memory object.
    mCentroids = clCreateBuffer(context, CL_MEM_READ_WRITE, centroid_size, NULL, &err);
    error(err);
    mPartitioned = clCreateBuffer(context, CL_MEM_READ_WRITE, partitioned_size, NULL, &err);
    error(err);
    mData = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
    error(err);
    mTmpCent = clCreateBuffer(context, CL_MEM_READ_WRITE, tmpcent_size, NULL, &err);
    error(err);
    mFinally = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Centroid_Data_t) * map_wg_size, NULL, &err);
    error(err); 
    //Iteration loop
    for(i=0; i < iteration_n; i++){
        //=============================Assignment step.===================================
        //Enqueue write data.
        err = clEnqueueWriteBuffer(q, mCentroids, CL_FALSE, 0, centroid_size, centroids, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(q, mPartitioned, CL_FALSE, 0, partitioned_size, partitioned, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(q, mData, CL_FALSE, 0, data_size, data, 0, NULL, &ev_buf);
        error(err);
        //Set kernel arg.
        err = clSetKernelArg(assign, 0, sizeof(cl_mem), &mCentroids);
        err |= clSetKernelArg(assign, 1, sizeof(cl_mem), &mPartitioned);
        err |= clSetKernelArg(assign, 2, sizeof(cl_mem), &mData);
        error(err);
        //Set lws, gws.
        assign_lws[0] = ASSIGN_LWS;
        assign_gws[0] = data_n;

        //Enqueue NDRange kernel.
        err =  clEnqueueNDRangeKernel(q, assign, 1, NULL, assign_gws, assign_lws, 1, &ev_buf, &ev_assign);
        error(err);

        //Enqueue Read data.
        err = clEnqueueReadBuffer(q, mPartitioned, CL_TRUE, 0, partitioned_size, partitioned, 1, &ev_assign, NULL); 
        error(err);    
        //===============================Update step.=====================================
        for(class_i = 0; class_i < class_n; class_i++){
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }
        //********************************* map ********************************************
        //Enqueue write data.
        err = clEnqueueWriteBuffer(q, mData, CL_FALSE, 0, data_size, data, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(q, mPartitioned, CL_FALSE, 0, partitioned_size, partitioned, 0, NULL, NULL);
        error(err);

        //Set kernel arg.
        err = clSetKernelArg(update_map, 0, sizeof(cl_mem), &mPartitioned);
        err |= clSetKernelArg(update_map, 1, sizeof(cl_mem), &mData);
        err |= clSetKernelArg(update_map, 2, sizeof(cl_mem), &mTmpCent);
        error(err);

        //Set lws, gws.
        update_lws[0] = LWS; //16
        update_lws[1] = class_n;
        update_gws[0] = data_div_class; //data_n / class_n
        update_gws[1] = class_n;

        //Enqueue NDRange kernel.
        err = clEnqueueNDRangeKernel(q, update_map, 2, NULL, update_gws, update_lws, 1, &ev_buf, &ev_update);
        error(err);

        //Enqueue Read data.
        err = clEnqueueReadBuffer(q, mTmpCent, CL_TRUE, 0, tmpcent_size, tmp_centroid, 1, &ev_update, NULL);
        error(err);

        //********************************* reduce ********************************************
        memset(finally, 0, sizeof(Centroid_Data_t) * map_wg_size);
        err = clEnqueueWriteBuffer(q, mTmpCent, CL_FALSE, 0, tmpcent_size, tmp_centroid, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(q, mFinally, CL_FALSE, 0, sizeof(Centroid_Data_t) * map_wg_size, finally, 0, NULL, &ev_buf); 
        error(err);

        err = clSetKernelArg(update_reduce, 0, sizeof(cl_mem), &mTmpCent);
        err |= clSetKernelArg(update_reduce, 1, sizeof(cl_mem), &mFinally);
        error(err);

        update_lws[1] = class_n; //16
        update_lws[0] = LWS;
        update_gws[1] = class_n; //data_n / class_n / LWS
        update_gws[0] = map_wg_size; //data_n / class_n / LWS

        err = clEnqueueNDRangeKernel(q, update_reduce, 2, NULL, update_gws, update_lws, 1, &ev_buf, &ev_update);
        error(err);

        err = clEnqueueReadBuffer(q, mFinally, CL_TRUE, 0, sizeof(Centroid_Data_t) * map_wg_size, finally, 1, &ev_update, NULL);
        error(err);
        
        for(class_i = 0; class_i < class_n; class_i++){
            for(data_i = 0; data_i < map_wg_size / LWS; data_i++){
                centroids[class_i].x += finally[class_i * (map_wg_size/LWS) + data_i].p.x; 
                centroids[class_i].y += finally[class_i * (map_wg_size/LWS) + data_i].p.y; 
                count[class_i] += finally[class_i * (map_wg_size/LWS) + data_i].cnt; 
            }
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }

    //Release.
    clReleaseMemObject(mData);
    clReleaseMemObject(mCentroids);
    clReleaseMemObject(mTmpCent);
    clReleaseMemObject(mPartitioned);

    clReleaseKernel(assign);
    clReleaseKernel(update_map);
    clReleaseKernel(update_reduce);
    clReleaseCommandQueue(q);
    clReleaseEvent(ev_buf);
    clReleaseEvent(ev_assign);
    clReleaseEvent(ev_update);
    clReleaseProgram(pg);
    clReleaseContext(context);

    free(count);
    free(tmp_centroid);
}

