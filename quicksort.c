#include "quicksort.h"
#include "pivot.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define NOPRINTING

int check_and_print(int *elements, int n, char *file_name){
    int sort_element=sorted_ascending(elements, n);
    if(!sort_element){
        printf("Error: the elements are not sorted in ascending order.\n");
    }
    FILE *file=fopen(file_name, "w");
    if (!file) return -1;

    for(int i=0; i<n; i++){
      fprintf(file, "%d", elements[i]);
      if(i < n-1) fprintf(file, " ");
    }
    fprintf(file, "\n");
    fclose(file);
    return 0;
}

int distribute_from_root(int *all_elements, int n, int **local_elements){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int base_elements=n/size;
    int remainder=n%size;
    int *counts=malloc(size*sizeof(int));
    int *chunk_start=malloc(size*sizeof(int));

    int offset=0;
    for(int i=0; i<size; i++){
        counts[i]=base_elements+(i<remainder? 1:0);
        chunk_start[i]=offset;
        offset+=counts[i];
    }

    int local_n=counts[rank];
    *local_elements=malloc(local_n*sizeof(int));
    MPI_Scatterv(all_elements, counts, chunk_start, MPI_INT, *local_elements, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    free(counts);
    free(chunk_start);

    return local_n;
}

void gather_on_root(int *all_elements, int *local_elements, int local_n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *counts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }

    MPI_Gather(&local_n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }

    MPI_Gatherv(local_elements, local_n, MPI_INT,
               all_elements, counts, displs, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(counts);
        free(displs);
    }
}

int global_sort(int **elements, int n, MPI_Comm comm, int pivot_strategy) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size == 1) {
        return n;
    }
    
    // We need to check if size is even before proceeding
    if (size % 2 != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Number of processes (%d) must be even at all recursion levels.\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    

     int actual_pivot_value;
    // We need the index of the first element greater than pivot_value
    int pivot_split_idx = select_pivot(pivot_strategy, *elements, n, comm, &actual_pivot_value);
    int half_size = size / 2;
    int new_color;
    int partner_rank;
    // here I spplit the data into send and keep two groups
    int send_n; 
    int keep_n; 
    int *send_ptr; 
    int *keep_ptr; 

    if (rank < half_size) { 
        new_color = 0;
        partner_rank = rank + half_size;

        // Here the Elements <= pivot_value are kept, elements > pivot_value are sent
        keep_n = pivot_split_idx;
        send_n = n - pivot_split_idx;
        keep_ptr = *elements; 
        send_ptr = *elements + pivot_split_idx; 

    } else { 
        new_color = 1;
        partner_rank = rank - half_size;

        // Elements > pivot_value are kept, elements <= pivot_value are sent
        keep_n = n - pivot_split_idx;
        send_n = pivot_split_idx;
        keep_ptr = *elements + pivot_split_idx;
        send_ptr = *elements; 
    }

    // Here I exchange sizes first
    int recv_n;
    MPI_Sendrecv(&send_n, 1, MPI_INT, partner_rank, 0,
                 &recv_n, 1, MPI_INT, partner_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    // Allocating receive buffer
    int* received_elements = (int*)malloc((recv_n > 0 ? recv_n : 1) * sizeof(int));
    if (recv_n > 0 && !received_elements) {
        fprintf(stderr, "Rank %d: Malloc for received_elements failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Here I exchange actual data
    MPI_Sendrecv(send_ptr, send_n, MPI_INT, partner_rank, 1,
                 received_elements, recv_n, MPI_INT, partner_rank, 1,
                 comm, MPI_STATUS_IGNORE);
   
    // Begin merge
    int new_n = keep_n + recv_n;
    int *merged_elements = (int*)malloc((new_n > 0 ? new_n : 1) * sizeof(int));
    if (new_n > 0 && !merged_elements) {
        fprintf(stderr, "Rank %d: Malloc for merged_elements failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    merge_ascending(keep_ptr, keep_n, received_elements, recv_n, merged_elements);
    
    
    if (received_elements) free(received_elements); 

    free(*elements); 
    *elements = merged_elements; 

    // Spliting communicator
    MPI_Comm new_comm;
    MPI_Comm_split(comm, new_color, rank, &new_comm);
    // Recursive call
    int result_n = global_sort(elements, new_n, new_comm, pivot_strategy);
    MPI_Comm_free(&new_comm);
    return result_n;
}
void merge_ascending(int *v1, int n1, int *v2, int n2, int *result){
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (v1[i] <= v2[j]) {
            result[k++] = v1[i++];
        } else {
            result[k++] = v2[j++];
        }
    }
    while (i < n1) {
        result[k++] = v1[i++];
    }
    while (j < n2) {
        result[k++] = v2[j++];
    }
}

int read_input(char *file_name, int **elements) {
    FILE *file = fopen(file_name, "r");
    if (!file) {
        perror("Couldn't open input file");
        return -1;
    }
    int num_values;
    if (fscanf(file, "%d", &num_values) != 1) {
        perror("Couldn't read element count from input file");
        fclose(file);
        return -1;
    }
    *elements = malloc(num_values * sizeof(int));

    if (!(*elements) && num_values > 0) {
        perror("Memory allocation failed");
        fclose(file);
        return -1;
    }

    for (int i = 0; i < num_values; i++) {
        if (fscanf(file, "%d", &((*elements)[i])) != 1) {
            perror("Couldn't read elements from input file");
            free(*elements);
            *elements = NULL;
            fclose(file);
            return -1;
        }
    }
    fclose(file);
    return num_values;
}

int sorted_ascending(int *elements, int n) {
    for (int i = 1; i < n; i++) {
        if (elements[i] < elements[i-1]) {
            printf("Error at index %d: %d > %d\n", i - 1, elements[i - 1], elements[i]);
            return 0;
        }
    }
    return 1;
}



void swap(int *e1, int *e2) {
    int tmp = *e1;
    *e1 = *e2;
    *e2 = tmp;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            printf("Usage: %s <input_file> <output_file> <pivot_strategy>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *input_name = argv[1];
    char *output_name = argv[2];
    int pivot_strategy = atoi(argv[3]);

    int* all_elements = NULL;
    int* local_elements = NULL;
    int total_n = 0;

    double overall_start_time = MPI_Wtime();
    
    if (rank == 0) {
        total_n = read_input(input_name, &all_elements);
        if (total_n <= 0) {
            if (all_elements) free(all_elements);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    MPI_Bcast(&total_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (total_n <= 0) {
        MPI_Finalize();
        return 1;
    }
    
    double distrubution_start_time = MPI_Wtime();
    int local_n = distribute_from_root(all_elements, total_n, &local_elements);
    double distrubution_end_time = MPI_Wtime();
    double current_distr_time = distrubution_end_time - distrubution_start_time;
    double local_serial_sort_start_time = MPI_Wtime();
    if (local_n > 1) { 
        qsort(local_elements, local_n, sizeof(int), compare); // 确保 compare 函数可用
    }
    double local_serial_sort_end_time = MPI_Wtime();
    double current_process_serial_time = local_serial_sort_end_time - local_serial_sort_start_time;
    MPI_Barrier ( MPI_COMM_WORLD );
    double global_sort_start_time = MPI_Wtime();
    int sorted_n = global_sort(&local_elements, local_n, MPI_COMM_WORLD, pivot_strategy);
    double global_sort_end_time = MPI_Wtime();
    double current_process_global_sort_time = global_sort_end_time - global_sort_start_time;
    MPI_Barrier ( MPI_COMM_WORLD );
    if (rank == 0) {
        free(all_elements);
    }
    all_elements = NULL;
    if (rank == 0) {
        all_elements = malloc(total_n * sizeof(int));
    }
    double gather_start_time=MPI_Wtime();
    gather_on_root(all_elements, local_elements, sorted_n);
    double gather_end_time=MPI_Wtime();
    double current_process_gather_time = gather_end_time - gather_start_time;

    double overall_end_time = MPI_Wtime();
    double current_process_overall_time = overall_end_time - overall_start_time;
    
    if (local_elements) free(local_elements);
 
    double max_distr_time;
    MPI_Reduce(&current_distr_time, &max_distr_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    double max_gather_time;
    MPI_Reduce(&current_process_gather_time, &max_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    
    double max_overall_time;
    MPI_Reduce(&current_process_overall_time, &max_overall_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_serial_time;
    MPI_Reduce(&current_process_serial_time, &max_serial_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_global_sort_time;
    MPI_Reduce(&current_process_global_sort_time, &max_global_sort_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        // printf("Initial Local Serial Sort (Max): %f seconds.\n", max_serial_time);
        // printf("distribution time (Max): %f seconds.\n", max_distr_time);
        // printf("Parallel Quicksort Phase (Max): %f seconds.\n", max_global_sort_time);
        // printf("gather on root time (Max): %f seconds.\n", max_gather_time);
        printf("%f\n", max_overall_time);
        
        check_and_print(all_elements, total_n, output_name);
        free(all_elements);
    }
    
    
    MPI_Finalize();
    return 0;
}
