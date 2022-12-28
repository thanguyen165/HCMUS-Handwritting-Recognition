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

def Flatten(matrix):
    arr = np.array(matrix)
    return arr.flatten()

def Average(matrix, r, c): # matrix - rows - columns
    arr = np.array(matrix)
    n = arr.shape[0]
    m = arr.shape[1]
    
    a = (n + r - 1) // r # caculate new number of rows
    b = (m + c - 1) // c # caculate new number of columns
    res = np.zeros((a, b))
    
    for i in range(n):
        for j in range(m):
            res[i // r][j // c] += arr[i][j]
            
    for i in range(a):
        for j in range(b):
            res[i][j] //= (r * c)
    
    return res

def Histogram(matrix):
    arr = np.array(matrix)
    res = np.zeros((256))
    
    for i in arr:
        for j in i:
            res[int(j)] += 1
            
    print(res)

#____________________________________________________________________#

X_train, y_train = load_mnist('data/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    #img = Average(img, 2, 2)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
