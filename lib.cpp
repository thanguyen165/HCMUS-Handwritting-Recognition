#include "lib.hpp"
#include <stdio.h>
#include <algorithm>

const int N = (int) 6e4, sz = 28 * 28;
node dis[N];

int guess(void * xav, void * yav, void * testav) {
    const int K = 1000;  // KNN
    // must be unsigned long long type
    unsigned long long * x = (unsigned long long *) xav;
    unsigned long long * y = (unsigned long long *) yav;
    unsigned long long * test = (unsigned long long *) testav;

    for (int i = 0; i < N; i++) {
        unsigned long long sum = 0;
        for (int j = 0; j < sz; j++) {
            unsigned long long a = x[i * sz + j], b = test[j];
            sum += (a - b)*(a - b);
        }
        dis[i].distance = sum;
        dis[i].label = y[i];
    }

    std::sort(dis, dis + N);

    int cnt[10];  // counting array
    for (int i = 0; i < 10; i++) cnt[i] = 0;
    for (int i = 0; i < K; i++) cnt[dis[i].label]++;
    
    int max_index = 0;
    for (int i = 0; i < 10; i++) {
        if (cnt[i] > cnt[max_index]) max_index = i;
    }
    return max_index;
}
