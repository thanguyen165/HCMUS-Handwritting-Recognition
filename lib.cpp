#include "lib.hpp"
#include <algorithm>

const int X_TRAIN_SIZE = (int) 6e4;
const int X_TEST_SIZE = (int) 1e4;
node dis[X_TRAIN_SIZE];
int cnt[10];

unsigned long long distance(void *a, void *b, int size) {
    unsigned long long *a_ = (unsigned long long *) a;
    unsigned long long *b_ = (unsigned long long *) b;

    unsigned long long ans = 0;
    for (int i = 0; i < size; i++) {
        ans += (a_[i] - b_[i])*(a_[i] - b_[i]);
    }
    return ans;
}

int guess_optimize(void *x_train_p, void *y_train_p, void *test_matrix_p, int img_size, int KNN) {
    unsigned long long * x_train = (unsigned long long *) x_train_p;
    unsigned long long * y_train = (unsigned long long *) y_train_p;
    unsigned long long * test = (unsigned long long *) test_matrix_p;

    for (int i = 0; i < X_TRAIN_SIZE; i++) {
        dis[i].distance = distance((x_train + i*img_size), test, img_size);
        dis[i].label = y_train[i];
    }

    std::sort(dis, dis + X_TRAIN_SIZE);

    for (int i = 0; i < 10; i++) cnt[i] = 0;
    for (int i = 0; i < KNN; i++) cnt[ dis[i].label ]++;
    
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (cnt[i] > cnt[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}
