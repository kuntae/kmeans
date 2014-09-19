typedef struct Point{
    float x;
    float y;
}Point;

typedef struct Package_t{
    float x;
    float y;
    int index;
}package_t;

typedef struct Centroid_Data_t{
    struct Point point;
    int cnt;
}Centroid_Data_t;

__kernel void assign(__global Point *centroids, __global int *partitioned, __global Point *data){
    int global_i = get_global_id(0);
    int local_i = get_local_id(0);
    int class_i;
    //local memory.
    __local float2 tmp_data[ASSIGN_LWS];
    tmp_data[local_i].x = data[global_i].x;
    tmp_data[local_i].y = data[global_i].y;

    __local float2 tmp_centroids[CLASS_N];
    if(local_i < CLASS_N){
        tmp_centroids[local_i].x = centroids[local_i].x;
        tmp_centroids[local_i].y = centroids[local_i].y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float2 t= (float2)(0.0);
    float dist, min_dist =DBL_MAX;

    for(class_i = 0; class_i < CLASS_N; class_i++){
        t = tmp_data[local_i] - tmp_centroids[class_i];
        dist = t.x * t.x + t.y * t.y;

        if(dist < min_dist){
            partitioned[global_i] = class_i;
            min_dist = dist;
        }
    }
    
}

__kernel void update_map(__global int *partitioned, __global Point *data, __global Centroid_Data_t *tmp_centroid){
    //global ID.
    int global_i = get_global_id(1);
    int global_j = get_global_id(0);
    //local ID
    int local_i = get_local_id(1);
    int local_j = get_local_id(0);
    
    int group_id = get_group_id(0);
    int tmp_count;
    Point p;

    //tmp 2X2 array of centriods.
    __local package_t package[CLASS_N][UPDATE_SIZE];
    
    //cast 2X2 array. 
    __global Point (*D)[DATA_N/CLASS_N] = (__global Point (*)[DATA_N/CLASS_N])data;
    __global int (*P)[DATA_N/CLASS_N] = (__global int (*)[DATA_N/CLASS_N])partitioned;
    __global Centroid_Data_t (*C)[WG] = (__global Centroid_Data_t (*)[WG])tmp_centroid;
    //local copy.    
    package[local_i][local_j].x = D[global_i][global_j].x;
    package[local_i][local_j].y = D[global_i][global_j].y;
    package[local_i][local_j].index = P[global_i][global_j];

    barrier(CLK_LOCAL_MEM_FENCE);

    int i, j;
    if(local_j == 0 && local_i < CLASS_N){
        p.x = 0.0; p.y = 0.0;
        tmp_count = 0;
        for(i = 0; i < CLASS_N; i++){
            for(j = 0; j < UPDATE_SIZE; j++){
                if(package[i][j].index == local_i){
                    p.x += package[i][j].x;
                    p.y += package[i][j].y;
                    tmp_count++;
                }
            }
        } 
        C[local_i][group_id].point.x = p.x;
        C[local_i][group_id].point.y = p.y;
        C[local_i][group_id].cnt = tmp_count;
    }
}

__kernel void update_reduce(__global Centroid_Data_t *tmp_centroid, __global Centroid_Data_t *finally){
    int g;
    //group info.
    int num_group = get_num_groups(0);
    int group_id = get_group_id(0);
    //global info.

    int local_size = get_local_size(0);
    //global id.
    int global_i = get_global_id(1);
    int global_j = get_global_id(0);
    
    int local_i = get_local_id(1);
    int local_j = get_local_id(0);
    //cast.
    __global Centroid_Data_t (*C)[WG] = (__global Centroid_Data_t (*)[WG])tmp_centroid;
    __global Centroid_Data_t (*F)[WG/UPDATE_SIZE] = (__global Centroid_Data_t (*)[WG/UPDATE_SIZE])finally;
    __local Centroid_Data_t mid[CLASS_N][UPDATE_SIZE];
    mid[local_i][local_j] = C[global_i][global_j];
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction.
    for(g = local_size >> 1; g >= 1; g = g >> 1){
        if(local_j < g && local_i < CLASS_N ){
            mid[local_i][local_j].point.x += mid[local_i][local_j+g].point.x;
            mid[local_i][local_j].point.y += mid[local_i][local_j+g].point.y;
            mid[local_i][local_j].cnt += mid[local_i][local_j+g].cnt;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
     
    if(local_j == 0 && local_i < CLASS_N){
        F[local_i][group_id].point.x = mid[local_i][local_j].point.x;
        F[local_i][group_id].point.y = mid[local_i][local_j].point.y;
        F[local_i][group_id].cnt = mid[local_i][local_j].cnt;
        //printf("#F[%d][%d] x %f y %f cnt %d\n", local_i, group_id, F[local_i][group_id].point.x, F[local_i][group_id].point.y, F[local_i][group_id].cnt); 
    }
}
