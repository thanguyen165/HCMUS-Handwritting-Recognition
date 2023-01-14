import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import ctypes

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

#----------------------------------------------------------------------------------------------------
# Functions
def Flatten(matrix):
    arr = np.array(matrix)
    return arr.flatten()

def prepare_flattened_array():
    for i in range(y_train.shape[0]):
        X_train_Flattened.append(Flatten(X_train[i]))

def Average(matrix, r, c): # matrix - number of rows to merge - number of columns to merge
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

def prepare_average_array(r = 2, c = 2):
    for i in range(y_train.shape[0]):
        X_train_Average.append(Flatten(Average(X_train[i], r, c)))

def Histogram(matrix):
    res = np.zeros((256))
    
    for i in matrix:
        for j in i:
            res[j] += 1
    return res

def prepare_histogram_array():
    for i in range(y_train.shape[0]):
        X_train_Histogram.append(Histogram(X_train[i]))

def Distance(a, b):
    ans = 0
    for i in range(a.shape[0]):
        ans += (a[i] - b[i]) * (a[i] - b[i])
    return ans

def guess(matrix, method = 1, KNN = 500, r = 2, c = 2):
    if method > 3 or method < 1:
        method = 1
    
    if method == 1:
        arr = Flatten(matrix)
        return lib.guess(ctypes.c_void_p(X_train_Flattened.ctypes.data), ctypes.c_void_p(y_train.ctypes.data), ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data))
    elif method == 2:
        arr = Flatten(Average(matrix, r, c))
        return lib.guess(ctypes.c_void_p(X_train_Average.ctypes.data), ctypes.c_void_p(y_train.ctypes.data), ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data))
    elif method == 3:
        arr = Histogram(matrix)
        return lib.guess(ctypes.c_void_p(X_train_Histogram.ctypes.data), ctypes.c_void_p(y_train.ctypes.data), ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data))
    
#-----------------------------------------------------------------------------------------------------------
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')
lib = ctypes.cdll.LoadLibrary('./lib.so')

print("prepare...")
method = 1
#method = 1 for flatten, 2 for average, 3 for histogram
KNN = 500
rows_to_ave = 2
columns_to_ave = 2

X_train_Flattened = []
prepare_flattened_array()
X_train_Average = []
prepare_average_array(rows_to_ave, columns_to_ave)
X_train_Histogram = []
# prepare_histogram_array()

print("done! let's rock")
print("_______")

X_train_Flattened = np.array(X_train_Flattened, dtype = np.uint0)
y_train = np.array(y_train, dtype = np.uint0)
y_test = np.array(y_test, dtype = np.uint0)


for i in range(10):
    print("Input is number %d" % y_test[i])
    print("computer guess: %d" % guess(X_test[i], method, KNN, rows_to_ave, columns_to_ave))
    print("----------------")

    
## below code is to show images
# print('MNIST train size: %d, img size: %d x %d' % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
# ax = ax.flatten()

# for i in range(10):
#     img = X_test[i]
#     ax[i].imshow(img, cmap='Blues', interpolation='nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()
