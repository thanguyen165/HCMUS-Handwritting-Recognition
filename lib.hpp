#ifndef LIB_H_INCLUDED
#define LIB_H_INCLUDED

#ifdef __cplusplus
   extern "C" {
#endif

struct node {
    int distance, label;

    bool operator < (const node& other) {
        return this->distance < other.distance;
    }
};

int guess_optimize(void *x_train_p, void *y_train_p, void *test_matrix_p, int img_size, int KNN);

#ifdef __cplusplus
   }
#endif

#endif /* LIB_H_INCLUDED */
