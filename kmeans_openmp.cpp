#include "kmeans.h"

#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>

#define NUM_THREAD 4
void kmeans(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned){
    // Loop indices for iteration, data and class
    int i, th_i, data_i, class_i;
    // Count number of data in each class
    int* count = (int*)malloc(sizeof(int) * class_n);
    // Temporal point value to calculate distance
    Point t;
    //temp variable for parallel.
    Point **tmp_centroids;
    int **tmp_count;

    //init temp variable.
    tmp_centroids = (Point**)malloc(sizeof(Point*) * NUM_THREAD);
    tmp_count = (int**)malloc(sizeof(int*) * NUM_THREAD);
    for(i = 0; i<NUM_THREAD; i++){
        tmp_centroids[i] = (Point*)calloc(class_n, sizeof(Point));
        tmp_count[i] = (int*)calloc(class_n, sizeof(int));
    }

    omp_set_num_threads(NUM_THREAD);
    // Iterate through number of interations
    for (i = 0; i < iteration_n; i++) {
        // Assignment step
#pragma omp parallel for private(data_i, class_i, t)
        for (data_i = 0; data_i < data_n; data_i++) {
            float min_dist = DBL_MAX;

            for (class_i = 0; class_i < class_n; class_i++) {
                t.x = data[data_i].x - centroids[class_i].x;
                t.y = data[data_i].y - centroids[class_i].y;

                float dist = t.x * t.x + t.y * t.y;

                if (dist < min_dist) {
                    partitioned[data_i] = class_i;
                    min_dist = dist;
                }
            }
        }

        // Update step
        // Clear sum buffer and class count
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x = 0.0;
            centroids[class_i].y = 0.0;
            count[class_i] = 0;
        }

        // Sum up and count data for each class
#pragma omp parallel for private(data_i)
        for (data_i = 0; data_i < data_n; data_i++) {         
            int num_th = omp_get_thread_num();
            tmp_centroids[num_th][partitioned[data_i]].x += data[data_i].x;
            tmp_centroids[num_th][partitioned[data_i]].y += data[data_i].y;
            tmp_count[num_th][partitioned[data_i]]++;
        }
        //reduction operation. 
        for(th_i = 0; th_i < NUM_THREAD; th_i++){
            for(class_i = 0; class_i < class_n; class_i+=2){
                centroids[class_i].x += tmp_centroids[th_i][class_i].x;
                centroids[class_i].y += tmp_centroids[th_i][class_i].y;
                count[class_i] += tmp_count[th_i][class_i];
                //unloop.
                centroids[class_i+1].x += tmp_centroids[th_i][class_i+1].x;
                centroids[class_i+1].y += tmp_centroids[th_i][class_i+1].y;
                count[class_i+1] += tmp_count[th_i][class_i+1];

                tmp_centroids[th_i][class_i].x = 0.0;
                tmp_centroids[th_i][class_i].y = 0.0;
                tmp_centroids[th_i][class_i+1].x = 0.0;
                tmp_centroids[th_i][class_i+1].y = 0.0;
                tmp_count[th_i][class_i] = 0;
                tmp_count[th_i][class_i+1] = 0;
            }
        }

        // Divide the sum with number of class for mean point
        for (class_i = 0; class_i < class_n; class_i++) {
            centroids[class_i].x /= count[class_i];
            centroids[class_i].y /= count[class_i];
        }
    }

    //free.
    for(i = 0; i < NUM_THREAD; i++){
        free(tmp_centroids[i]);
        free(tmp_count[i]);
    }
}
