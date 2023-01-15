import matplotlib.pyplot as plt
import os
import numpy as np
import gzip

def load_mnist(path, kind='train'):
    """Load MNIST data from 'path' """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype = np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype = np.uint8).reshape(len(labels), 28, 28).astype(np.float64)

    return images, labels

#------------------------------------------------------------------------------------
# Functions
def distance(a, b):
    ans = 0
    for i in range(a.shape[0]):
        ans += (a[i] - b[i]) * (a[i] - b[i])
    return ans

def Flatten(matrix):
    arr = np.array(matrix)
    return arr.flatten()

def Average(matrix, r = 2, c = 2): # matrix - number of rows to merge - number of columns to merge
    n = matrix.shape[0]
    m = matrix.shape[1]
    
    a = (n + r - 1) // r   # new number of columns
    b = (m + c - 1) // c   # new number of rows
    res = np.zeros((a, b))
    
    for i in range(n):
        for j in range(m):
            res[i // r][j // c] += matrix[i][j]
            
    for i in range(a):
        for j in range(b):
            res[i][j] //= (r * c)
    return res

def Histogram(matrix):
    res = np.zeros((256))
    for i in matrix:
        for j in i:
            res[int(j)] += 1
    return res

#----------------------
def guess(matrix, method = 1, KNN = 500, r = 2, c = 2):
    if method > 3 or method < 1:
        method = 1
    
    if method == 1:
        arr = Flatten(matrix)
    elif method == 2:
        arr = Flatten(Average(matrix, r, c))
    elif method == 3:
        arr = Histogram(matrix)
    
    dis = np.zeros(y_train.shape[0])
    
    for i in range(y_train.shape[0]):
        if method == 1:
            dis[i] = distance(X_train_Flattened[i], arr)
        elif method == 2:
            dis[i] = distance(X_train_Average[i], arr)
        elif method == 3:
            dis[i] = distance(X_train_Histogram[i], arr)

    index = np.lexsort((y_train, dis))  # Sort by dis, then by y_train
    
    K = KNN   # k nearest neighbours
    cnt = np.zeros(10)  # counting array
    for i in range(K):
        cnt[ y_train[index[i]] ] += 1
        
    max_index = 0
    for i in range(10):
        if cnt[i] > cnt[max_index]:
            max_index = i
    return max_index

#----------------------
def cal_accuracy(method = 1, KNN = 500):
    cnt = y_test.shape[0]
    cnt = 100
    true_ = 0
    for i in range(cnt):
        true_ += (y_test[i] == guess(X_test[i], method, KNN, rows_to_ave, columns_to_ave))
    return true_ / cnt

#-----------------------------------------------------------------------------------------------------------
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')

print("prepare...")
rows_to_ave = 2
columns_to_ave = 2

X_train_Flattened = [Flatten(i) for i in X_train]
X_train_Average = [Flatten(Average(i, rows_to_ave, columns_to_ave)) for i in X_train]
X_train_Histogram = [Histogram(i) for i in X_train]

print("done! Let's rock")
print("__________________")

KNNarray = [10, 100, 500]
methodarray = [1, 2, 3]
ans = np.zeros((4))

for KNN in KNNarray:
    for method in methodarray:
        ans[method] = cal_accuracy(method, KNN)
    print("K = %d" % KNN)
    print("    Flatten: %f%%" % (ans[1] * 100))
    print("    Average: %f%%" % (ans[2] * 100))
    print("    Histogram: %f%%" % (ans[3] * 100))
    print("------------------------------")

## below code allows us to guess the number written in img.
# print("Computer guess your number is %d" % guess(img, method, KNN, rows_to_ave, columns_to_ave))

## below code is to show images
# print('MNIST train size: %d, img size: %d x %d' % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()

# for i in range(5):
#     img = X_test[i]
#     ax[i].imshow(img, cmap='Blues', interpolation='nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
