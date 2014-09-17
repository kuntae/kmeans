#include "kmeans.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <CL/cl.h>

#define error(err) if(err !=CL_SUCCESS){fprintf(stderr, "[%s:%d] error: %d\n",__FILE__, __LINE__, err);exit(EXIT_FAILURE);}
#define MAX_DEVICE 4
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
    cl_device_id *dev;
    cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
    cl_context context;
    cl_command_queue *queue;
    cl_kernel *assign, *update_map, *update_reduce;
    cl_program pg;
    cl_int err;
    cl_event *ev_write, *ev_read, *ev_assign, *ev_update;
    size_t assign_lws[1], assign_gws[1];
    size_t update_lws[2], update_gws[2];

    //loop variable.
    int i, iteration_i, data_i, class_i;
    
    //Nomal variable.
    size_t data_size, centroid_size, partitioned_size, tmpcent_size, finally_size;
    int dev_data_n, data_div_class_n, tmp_cent_n, finally_n;

    //count number of data in each class
    int *count = (int*)malloc(class_n* sizeof(int));
    if(count == NULL){
        fprintf(stderr, "%d line allocate error\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    //==========================================OpenCL========================================
    //Get platform id.
    err = clGetPlatformIDs(1, &plat, NULL);
    error(err);

    //Get device id.
    cl_uint num_dev;
    err = clGetDeviceIDs(plat, dev_type, 0, NULL, &num_dev);
    error(err);
    if(num_dev < 0){
        fprintf(stderr, "This platform is not match device type.(%s, %d)",__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    dev = (cl_device_id*)malloc(sizeof(cl_device_id) * num_dev);
    err = clGetDeviceIDs(plat, dev_type, num_dev, dev, NULL);
    error(err);

    //len.
    dev_data_n = data_n / num_dev;
    data_div_class_n = dev_data_n / class_n;
    tmp_cent_n = data_div_class_n / LWS;
    finally_n = tmp_cent_n / LWS;

    //byte size. 
    data_size = sizeof(Point) * dev_data_n;
    centroid_size = sizeof(Point) * class_n;
    partitioned_size = sizeof(int) * dev_data_n;
    tmpcent_size = sizeof(Centroid_Data_t) * tmp_cent_n  * class_n;
    finally_size = sizeof(Centroid_Data_t) * finally_n * class_n;

    //malloc() tmp_centroid.
    Centroid_Data_t *tmp_centroid = (Centroid_Data_t*)malloc(tmpcent_size * num_dev);
    if(tmp_centroid == NULL){
        fprintf(stderr, "%d line allocate error\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    //malloc() finally. 
    Centroid_Data_t *finally = (Centroid_Data_t*)malloc(finally_size * num_dev);
    if(finally == NULL){
        fprintf(stderr, "%d line allocate error\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    //Create context.
    context = clCreateContext(NULL, num_dev, dev, NULL, NULL, &err);
    error(err);

    //Create command queue.
    queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_dev);
    for(i = 0; i < (int)num_dev; i++){
        queue[i] = clCreateCommandQueue(context, dev[i], 0, &err);
        error(err);
    }

    //Create a program.
    size_t len;
    char *src;
    FILE *fp = fopen("kmeans_kernel.cl", "r");
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
            class_n, data_n, tmp_cent_n, ASSIGN_LWS, LWS);
    err = clBuildProgram(pg, num_dev, dev, option, NULL, NULL);
    if(err != CL_SUCCESS){
        char *log;
        clGetProgramBuildInfo(pg, dev[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        log = (char*)malloc(sizeof(char) * len +1);
        clGetProgramBuildInfo(pg, dev[0], CL_PROGRAM_BUILD_LOG, len, log, NULL);
        log[len] = '\0';
        fprintf(stderr, "===================================================");
        fprintf(stderr, "%s\n", log);
        fprintf(stderr, "===================================================");
        free(log);
        exit(EXIT_FAILURE);
    }
    assign = (cl_kernel*)malloc(sizeof(cl_kernel) * num_dev);
    update_map = (cl_kernel*)malloc(sizeof(cl_kernel) * num_dev);
    update_reduce = (cl_kernel*)malloc(sizeof(cl_kernel) * num_dev);
    //Create kernel.
    for(i = 0; i < (int)num_dev; i++){
        assign[i] = clCreateKernel(pg, "assign", &err);
        error(err);
        update_map[i] = clCreateKernel(pg, "update_map", &err);
        error(err);
        update_reduce[i] = clCreateKernel(pg, "update_reduce", &err);
        error(err);
    }
    //Variable for buffer object.
    cl_mem *mCentroids, *mPartitioned, *mData, *mTmpCent, *mFinally;

    mCentroids = (cl_mem*)malloc(sizeof(cl_mem) * num_dev);
    mPartitioned = (cl_mem*)malloc(sizeof(cl_mem) * num_dev);
    mData = (cl_mem*)malloc(sizeof(cl_mem) * num_dev);
    mTmpCent = (cl_mem*)malloc(sizeof(cl_mem) * num_dev);
    mFinally = (cl_mem*)malloc(sizeof(cl_mem) * num_dev);

    //Create memory object.
    for(i = 0; i < (int)num_dev; i++){
        mCentroids[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, centroid_size, NULL, &err);
        error(err);
        mPartitioned[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, partitioned_size, NULL, &err);
        error(err);
        mData[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &err);
        error(err);
        mTmpCent[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, tmpcent_size, NULL, &err);
        error(err);
        mFinally[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, finally_size, NULL, &err);
        error(err); 
    }

    //Allocate cl_event array.
    ev_write = (cl_event*)malloc(sizeof(cl_event) * num_dev); 
    ev_read = (cl_event*)malloc(sizeof(cl_event) * num_dev); 
    ev_assign = (cl_event*)malloc(sizeof(cl_event) * num_dev); 
    ev_update = (cl_event*)malloc(sizeof(cl_event) * num_dev); 

    //Iteration loop
    for(iteration_i=0; iteration_i < iteration_n; iteration_i++){
        //=============================Assignment step.===================================
        assign_lws[0] = ASSIGN_LWS; //256
        assign_gws[0] = dev_data_n; //data_n / num_dev;
        //Enqueue write data.
        for(i = 0; i < (int)num_dev; i++){
            err = clEnqueueWriteBuffer(queue[i], mCentroids[i], CL_FALSE, 0, centroid_size, centroids, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue[i], mPartitioned[i], CL_FALSE, 0, partitioned_size, &partitioned[i * dev_data_n], 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue[i], mData[i], CL_FALSE, 0, data_size, &data[i * dev_data_n], 0, NULL, &ev_write[i]);
            error(err);

            //Set kernel arg.
            err = clSetKernelArg(assign[i], 0, sizeof(cl_mem), &mCentroids[i]);
            err |= clSetKernelArg(assign[i], 1, sizeof(cl_mem), &mPartitioned[i]);
            err |= clSetKernelArg(assign[i], 2, sizeof(cl_mem), &mData[i]);
            error(err);

            //Enqueue NDRange kernel.
            err =  clEnqueueNDRangeKernel(queue[i], assign[i], 1, NULL, assign_gws, assign_lws, 1, &ev_write[i], &ev_assign[i]);
            error(err);

            //Enqueue Read data.
            err = clEnqueueReadBuffer(queue[i], mPartitioned[i], CL_FALSE, 0, partitioned_size, &partitioned[i * dev_data_n], 1, &ev_assign[i], &ev_read[i]); 
            error(err);    
        }
        //Wait read buffer.
        clWaitForEvents(num_dev, ev_read);

        //===============================Update step.=====================================
        for(class_i = 0; class_i < class_n; class_i++){
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }

        //********************************* map ********************************************
        //Set lws, gws.
            update_lws[0] = LWS; //16
            update_lws[1] = class_n;
            update_gws[0] = data_div_class_n; //data_n/num_dev / class_n
            update_gws[1] = class_n;

        for(i = 0; i < (int)num_dev; i++){
            //Enqueue write data.
            err = clEnqueueWriteBuffer(queue[i], mData[i], CL_FALSE, 0, data_size, &data[i * dev_data_n], 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue[i], mPartitioned[i], CL_FALSE, 0, partitioned_size, &partitioned[i * dev_data_n], 0, NULL, &ev_write[i]);
            error(err);

            //Set kernel arg.
            err = clSetKernelArg(update_map[i], 0, sizeof(cl_mem), &mPartitioned[i]);
            err |= clSetKernelArg(update_map[i], 1, sizeof(cl_mem), &mData[i]);
            err |= clSetKernelArg(update_map[i], 2, sizeof(cl_mem), &mTmpCent[i]);
            error(err);

            //Enqueue NDRange kernel.
            err = clEnqueueNDRangeKernel(queue[i], update_map[i], 2, NULL, update_gws, update_lws, 1, &ev_write[i], &ev_update[i]);
            error(err);

            //Enqueue Read data.
            err = clEnqueueReadBuffer(queue[i], mTmpCent[i], CL_FALSE, 0, tmpcent_size, &tmp_centroid[i * tmp_cent_n * class_n], 1, &ev_update[i], &ev_read[i]);
            error(err);
        }

        clWaitForEvents(num_dev, ev_read);

        //********************************* reduce ********************************************
        //finally array init.
        memset(finally, 0, finally_size * num_dev);

        update_lws[1] = LWS; //16
        update_lws[0] = class_n;
        update_gws[1] = tmp_cent_n; //data_n / class_n / LWS
        update_gws[0] = class_n; //data_n / class_n / LWS

        for(i = 0; i < (int)num_dev; i++){
            err = clEnqueueWriteBuffer(queue[i], mTmpCent[i], CL_FALSE, 0, tmpcent_size, &tmp_centroid[i * tmp_cent_n * class_n], 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(queue[i], mFinally[i], CL_FALSE, 0, finally_size, &finally[i * finally_n * class_n], 0, NULL, &ev_write[i]); 
            error(err);

            err = clSetKernelArg(update_reduce[i], 0, sizeof(cl_mem), &mTmpCent[i]);
            err |= clSetKernelArg(update_reduce[i], 1, sizeof(cl_mem), &mFinally[i]);
            error(err);

            err = clEnqueueNDRangeKernel(queue[i], update_reduce[i], 2, NULL, update_gws, update_lws, 1, &ev_write[i], &ev_update[i]);
            error(err);

            err = clEnqueueReadBuffer(queue[i], mFinally[i], CL_FALSE, 0, finally_size, &finally[i * finally_n * class_n], 1, &ev_update[i], &ev_read[i]);
            error(err);

        }

        clWaitForEvents(num_dev, ev_read);

        for(class_i = 0; class_i < class_n; class_i++){
            for(data_i = 0; data_i < finally_n * (int)num_dev ; data_i++){
                centroids[class_i].x += finally[data_i * class_n + class_i].p.x; 
                centroids[class_i].y += finally[data_i * class_n + class_i].p.y; 
                count[class_i] += finally[data_i * class_n  + class_i].cnt; 
            }
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }

    //Release.
    for(i = 0; i < (int)num_dev; i++){
        clReleaseMemObject(mData[i]);
        clReleaseMemObject(mCentroids[i]);
        clReleaseMemObject(mTmpCent[i]);
        clReleaseMemObject(mPartitioned[i]);
        clReleaseKernel(assign[i]);
        clReleaseKernel(update_map[i]);
        clReleaseKernel(update_reduce[i]);
        clReleaseCommandQueue(queue[i]);
        clReleaseEvent(ev_write[i]);
        clReleaseEvent(ev_read[i]);
        clReleaseEvent(ev_assign[i]);
        clReleaseEvent(ev_update[i]);
    }

    clReleaseProgram(pg);
    clReleaseContext(context);

    free(count);
    free(tmp_centroid);
    free(finally);
}

