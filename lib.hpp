#ifndef LIB1_H_INCLUDED
#define LIB1_H_INCLUDED

#ifdef __cplusplus
   extern "C" {
#endif

struct node {
    int distance, label;

    bool operator < (const node& other) {
        return this->distance < other.distance;
    }
};

int guess(void * xav, void * yav, void * testav);

#ifdef __cplusplus
   }
#endif

#endif /* LIB1_H_INCLUDED */
