#include "pivot.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

int compare(const void* v1, const void* v2){
    return (*(int*)v1-*(int*)v2);
}

int get_median(int* elements, int n) {
    if (n == 0) return 0;
    if (n % 2 == 0) {
        return elements[n / 2 - 1];
    } else {
        return elements[n / 2];
    }
}

int get_larger_index(int *elements, int n, int val) {
    for (int i = 0; i < n; i++) {
        if (elements[i] > val) return i;
    }
    return n;
}

int select_pivot_median_root(int *elements, int n, MPI_Comm comm, int *pivot_value) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int pivot_val = 0;

    if (rank == 0) {
        if (n > 0) {
            pivot_val = get_median(elements, n);
        }
    }
    MPI_Bcast(&pivot_val, 1, MPI_INT, 0, comm);

    *pivot_value = pivot_val;
    return get_larger_index(elements, n, pivot_val);
}

int select_pivot_mean_median(int *elements, int n, MPI_Comm comm, int *pivot_value) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_median = 0;
    int has_elements = (n > 0) ? 1 : 0;
    if (n > 0) {
        local_median = get_median(elements, n);
    }

    int* all_medians = NULL;
    int* has_elements_array = NULL;
    if (rank == 0) {
        all_medians = (int*)malloc(size * sizeof(int));
        has_elements_array = (int*)malloc(size * sizeof(int));
        if (!all_medians || !has_elements_array) {
            perror("Rank 0: Malloc failed in select_pivot_mean_median");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_median, 1, MPI_INT, all_medians, 1, MPI_INT, 0, comm);
    MPI_Gather(&has_elements, 1, MPI_INT, has_elements_array, 1, MPI_INT, 0, comm);

    int pivot_val = 0;
    if (rank == 0) {
        long long sum = 0;
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (has_elements_array[i]) {
                sum += all_medians[i];
                count++;
            }
        }
        pivot_val = (count > 0) ? (int)(sum / count) : 0;
        free(all_medians);
        free(has_elements_array);
    }
    MPI_Bcast(&pivot_val, 1, MPI_INT, 0, comm);

    *pivot_value = pivot_val;
    return get_larger_index(elements, n, pivot_val);
}

int select_pivot_median_median(int *elements, int n, MPI_Comm comm, int *pivot_value) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_median = 0;
    int has_elements = (n > 0) ? 1 : 0;
    if (n > 0) {
        local_median = get_median(elements, n);
    }

    int* all_medians = NULL;
    int* has_elements_array = NULL;
    if (rank == 0) {
        all_medians = (int*)malloc(size * sizeof(int));
        has_elements_array = (int*)malloc(size * sizeof(int));
        if (!all_medians || !has_elements_array) {
            perror("Rank 0: Malloc failed in select_pivot_median_median");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&local_median, 1, MPI_INT, all_medians, 1, MPI_INT, 0, comm);
    MPI_Gather(&has_elements, 1, MPI_INT, has_elements_array, 1, MPI_INT, 0, comm);

    int pivot_val = 0;
    if (rank == 0) {
        int* valid_medians = (int*)malloc(size * sizeof(int));
        if (!valid_medians) {
            perror("Rank 0: Malloc failed for valid_medians");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int valid_count = 0;

        for (int i = 0; i < size; i++) {
            if (has_elements_array[i]) {
                valid_medians[valid_count++] = all_medians[i];
            }
        }

        if (valid_count > 0) {
            qsort(valid_medians, valid_count, sizeof(int), compare);
            pivot_val = get_median(valid_medians, valid_count);
        }

        free(all_medians);
        free(has_elements_array);
        free(valid_medians);
    }
    MPI_Bcast(&pivot_val, 1, MPI_INT, 0, comm);

    *pivot_value = pivot_val;
    return get_larger_index(elements, n, pivot_val);
}

int select_pivot_smallest_root(int *elements, int n, MPI_Comm comm, int *pivot_value) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    int pivot_val = 0;

    if (rank == 0 && n > 0) {
        pivot_val = elements[0];
    }
    MPI_Bcast(&pivot_val, 1, MPI_INT, 0, comm);

    *pivot_value = pivot_val;
    return get_larger_index(elements, n, pivot_val);
}

int select_pivot(int pivot_strategy, int *elements, int n, MPI_Comm communicator, int *pivot_value) {
    int pivot_index_result = 0;

    switch (pivot_strategy) {
        case MEDIAN_ROOT:
            pivot_index_result = select_pivot_median_root(elements, n, communicator, pivot_value);
            break;
        case MEAN_MEDIAN:
            pivot_index_result = select_pivot_mean_median(elements, n, communicator, pivot_value);
            break;
        case MEDIAN_MEDIAN:
            pivot_index_result = select_pivot_median_median(elements, n, communicator, pivot_value);
            break;
        default: // SMALL_ROOT
            pivot_index_result = select_pivot_smallest_root(elements, n, communicator, pivot_value);
            break;
    }
    return pivot_index_result;
}

