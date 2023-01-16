import matplotlib.pyplot as plt
import os
import numpy as np
import gzip
import ctypes
import cv2
import time

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

##_________________________________________________________________________
## Functions
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

##----------------------
def guess(matrix, method = 1, KNN = 500, r = 2, c = 2):
    if method > 3 or method < 1:
        method = 1
    
    if method == 1:
        arr = Flatten(matrix)

        return lib.guess_optimize(
            ctypes.c_void_p(X_train_Flattened.ctypes.data),
            ctypes.c_void_p(y_train.ctypes.data),
            ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data),
            arr.shape[0],
            KNN
        )
    elif method == 2:
        arr = Flatten(Average(matrix, r, c))

        return lib.guess_optimize(
            ctypes.c_void_p(X_train_Average.ctypes.data),
            ctypes.c_void_p(y_train.ctypes.data),
            ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data),
            arr.shape[0],
            KNN
        )
    elif method == 3:
        arr = Histogram(matrix)

        return lib.guess_optimize(
            ctypes.c_void_p(X_train_Histogram.ctypes.data),
            ctypes.c_void_p(y_train.ctypes.data),
            ctypes.c_void_p((np.array(arr, dtype = np.uint0)).ctypes.data),
            arr.shape[0],
            KNN
        )
  
##----------------------
def cal_accuracy():
    KNNarray = [10, 100, 500]
    methodarray = [1, 2, 3]
    ans = np.zeros((4))

    for KNN in KNNarray:
        for method in methodarray:
            cnt = y_test.shape[0]
            count_true = 0
            for i in range(cnt):
                count_true += (y_test[i] == guess(X_test[i], method, KNN, rows_to_ave, columns_to_ave))
            ans[method] = count_true / cnt

        print("K = %d" % KNN)
        print("    Flatten: %f%%" % (ans[1] * 100))
        print("    Average: %f%%" % (ans[2] * 100))
        print("    Histogram: %f%%" % (ans[3] * 100))
        print("------------------------------")

##----------------------
def import_image(img_path):
    res = cv2.imread(img_path)
    img_column_size, img_row_size = res.shape[0], res.shape[1]
    img = np.zeros((img_column_size, img_row_size))
    if res[0][0][1] < 10:
        for i in range(img_column_size):
            for j in range(img_row_size):
                img[i][j] = res[i][j][1]
                if img[i][j] < 170:
                    img[i][j] = 0
                else:
                    img[i][j] = 256
    else:
        for i in range(img_column_size):
            for j in range(img_row_size):
                img[i][j] = 255 - res[i][j][1]
                if img[i][j] < 170:
                    img[i][j] = 0
                else:
                    img[i][j] = 256
    img = Average(img, img_column_size // 28 + (img_column_size % 28 != 0), img_row_size // 28 + (img_row_size % 28 != 0))
    return img

#-----------------------------------------------------------------------------------------------------------
print("prepare...")
time_begin = time.time()

X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')
lib = ctypes.cdll.LoadLibrary('./lib.so')

rows_to_ave = 2
columns_to_ave = 2

X_train_Flattened = [Flatten(i) for i in X_train]
X_train_Average = [Flatten(Average(i, rows_to_ave, columns_to_ave)) for i in X_train]
X_train_Histogram = [Histogram(i) for i in X_train]

X_train_Flattened = np.array(X_train_Flattened, dtype = np.uint0)
X_train_Average = np.array(X_train_Average, dtype = np.uint0)
X_train_Histogram = np.array(X_train_Histogram, dtype = np.uint0)
X_test = np.array(X_test, dtype = np.uint0)
y_train = np.array(y_train, dtype = np.uint0)
y_test = np.array(y_test, dtype = np.uint0)

time_end_loading = time.time()

print("Done! Time for loading: ", time_end_loading - time_begin)
print("_______________________________")

cal_accuracy()
time_end = time.time()
print("\nTime for running: ", time_end_loading - time_begin)
print("\nTotal time: ", time_end - time_begin)

##______________________________________________________________
## below code allows us to guess the number written in img.
# method = 1
# # method = 1 for flatten, 2 for average, 3 for histogram
# KNN = 500
# for i in range(90, 95):
#     print("Input is number %d" % y_test[i])
#     print("Computer guess your number is %d" % guess(X_test[i], method, KNN, rows_to_ave, columns_to_ave))
#     print("----------------")


##______________________________________________________________
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
