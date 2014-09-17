#include "kmeans.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define NUM_THREAD 4

typedef struct assign_data_t{
    int class_n;
    int data_n;
    Point *centroids;
    Point *data;
    int *partitioned;
}assign_data_t;

typedef struct update_data_t{
    int class_n;
    int data_n;
    Point *data;
    Point *centroids;
    int *partitioned;
    int *count;
    pthread_mutex_t *mutex;
}update_data_t;

void* assignment(void *assign_data_t);
void* update(void *update_data_t);

void kmeans(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned){
    printf("# Pthread.\n");
    int i, iter_i;
    //pthread variable.
    assign_data_t assign_data[NUM_THREAD];
    update_data_t update_data[NUM_THREAD];
    pthread_t assign_th[NUM_THREAD], update_th[NUM_THREAD];
    pthread_mutex_t mutex;
    int *count;

    count = (int*)calloc(class_n, sizeof(int));

    pthread_mutex_init(&mutex, NULL);
    //assignment step.
    for(iter_i=0; iter_i<iteration_n; iter_i++){
        //pthread function arg init.
        for(i = 0; i<NUM_THREAD; i++){
            assign_data[i].class_n = class_n;
            assign_data[i].data_n = data_n/NUM_THREAD;
            assign_data[i].centroids = centroids;
            assign_data[i].data = &data[i * assign_data[i].data_n];
            assign_data[i].partitioned = &partitioned[i * assign_data[i].data_n];
        }
        //init
        for(i=0; i<NUM_THREAD; i++){
            pthread_create(&assign_th[i], NULL, assignment, &assign_data[i]);

        }
        for(i=0; i<NUM_THREAD; i++){
            pthread_join(assign_th[i], NULL);
        }

        //clear sum buffer and class count.
        for(i = 0; i < class_n; i++){
            centroids[i].x = 0.0;
            centroids[i].y = 0.0;
            count[i] = 0;
        }

        //update step.
        for(i=0; i<NUM_THREAD; i++){
            update_data[i].class_n = class_n;
            update_data[i].data_n = data_n/NUM_THREAD;
            update_data[i].data = &data[i * update_data[i].data_n];
            update_data[i].centroids = centroids;
            update_data[i].partitioned = &partitioned[i * update_data[i].data_n];
            update_data[i].count = count;
            update_data[i].mutex = &mutex;
        }

        for(i=0; i<NUM_THREAD; i++){
            pthread_create(&update_th[i], NULL, update, &update_data[i]);
        }
        for(i=0; i<NUM_THREAD; i++){
            pthread_join(update_th[i], NULL);
        }

        // Divide the sum with number of class for mean point
        for (i = 0; i < class_n; i++) {
            centroids[i].x /= count[i];
            centroids[i].y /= count[i];
        }
    }
    pthread_mutex_destroy(&mutex);
}

void* assignment(void *assign_data){
    //cast arg.
    assign_data_t *tmp = (assign_data_t*)assign_data;
    //init variable.
    int class_n = tmp->class_n;
    int data_n = tmp->data_n;
    Point *centroids = tmp->centroids;
    Point *data = tmp->data;
    int *partitioned = tmp->partitioned;
    //loop variable. 
    int i, j;
    //calc variable.
    float dist;
    float min_dist;
    Point t;

    for(i = 0; i < data_n; i++){
        min_dist = DBL_MAX;
        for(j = 0; j < class_n; j++){
            t.x = data[i].x - centroids[j].x;
            t.y = data[i].y - centroids[j].y;

            dist = t.x * t.x + t.y * t.y;

            if(dist < min_dist){
                partitioned[i] = j;
                min_dist = dist;
            }
        }
    }
    return NULL;
}

void* update(void *update_data){
    //cast arg.
    update_data_t *tmp = (update_data_t*)update_data;
    //init variable.
    int class_n = tmp->class_n;
    int data_n = tmp->data_n;
    Point *centroids = tmp->centroids;
    Point *data = tmp->data;
    int *partitioned = tmp->partitioned;
    int *count = tmp->count;
    pthread_mutex_t *mu = tmp->mutex;
    //loop variable.
    int i;
    Point *t;
    int *c;

    t = (Point*)calloc(class_n,sizeof(Point));
    c = (int*)calloc(class_n, sizeof(int));

    for(i=0; i<data_n; i++){
       t[partitioned[i]].x +=data[i].x;
       t[partitioned[i]].y +=data[i].y;
       c[partitioned[i]]++;
    }

    pthread_mutex_lock(mu);
    for(i = 0; i<class_n; i++){
        centroids[i].x +=t[i].x;
        centroids[i].y +=t[i].y;
        count[i] +=c[i];
    }
    pthread_mutex_unlock(mu);

    free(t);
    free(c);
    return NULL;
}





