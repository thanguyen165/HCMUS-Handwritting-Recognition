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

#----------------------------------------------------------------------------------------------------
# Functions

def Flatten(matrix):
    arr = np.array(matrix)
    return arr.flatten()

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

def Histogram(arr):
    res = np.zeros((256))
    
    for i in arr:
        for j in i:
            res[j] += 1
    return res

def Distance(a, b):
    ans = 0
    for i in range(a.shape[0]):
        ans += (a[i] - b[i]) * (a[i] - b[i])
    return ans

def Guess(matrix):
    K = 500   # KNN
    dis = np.zeros(y_train.shape[0])
    
    arr = Flatten(matrix)
    for i in range(y_train.shape[0]):
        dis[i] = Distance(X_train_Flattened[i], arr)
        
    index = np.lexsort((y_train, dis))  # Sort by dis, then by y_train
    
    cnt = np.zeros(10)  # counting array
    for i in range(K):
        cnt[ y_train[index[i]] ] += 1
        
    max_index = 0
    for i in range(10):
        if cnt[i] > cnt[max_index]:
            max_index = i
    return max_index
    
#-----------------------------------------------------------------------------------------------------------
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='t10k')

X_train_Flattened = []
for i in range(y_train.shape[0]):
    X_train_Flattened.append(Flatten(X_train[i]))

for i in range(10):
    print("Input is number %d" % y_test[i])
    print("Computer guess your number is %d" % Guess(X_test[i]))
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
