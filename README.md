# HCMUS-Handwritting-Recognition

## 1/ Authors:
This is our final project for Introduction to Information Technology.

Our team:

1. [Phuc, Song Dong Gia](https://github.com/fusodoya)

2. [Loi, Nguyen Minh](https://github.com/mf0212/)

3. [Thang, Nguyen Quang](https://github.com/thanguyen165/)

## 2/ Environment
### 2.1. Python 3.7
Download at https://docs.conda.io/en/latest/miniconda.html
### 2.2. Visual Studio Code (VS Code):
Download at https://code.visualstudio.com/Download/

#### Install extension "Python" after installing VS Code.

### 2.3. Numpy Library:
``` pip install numpy```
### 2.4. Matplotlib Library:
``` pip install matplotlib```
### 2.5. cv2 Library
``` pip install opencv-python```

## 3/ Prepare MNIST dataset
Download MNIST dataset at: http://yann.lecun.com/exdb/mnist/ and **DO NOT UNZIP FILES**.

The MNIST dataset contains 60,000 images used to recognise input numbers called ```train```, and 10,000 images used to check if the algorithm is good or bad, called ```test```. Every image has its label, respective to the number written in the image.

## 4/ Organise project
4 zips of MNIST dataset is in ```data``` subfolder.

## 5/ Before we start...
Run ```test_MNIST.py``` file to make sure MNIST dataset is successfully installed and set up.

## 6/ How do we recognise the numbers?
* Step 1: Vectorize all the images of ```train``` dataset and the ```input img```.

* Step 2: Find the **distance** between ```input img``` and each img in ```train```.

* Step 3: Sort all the distances in increasing order.

* Step 4: Choose ```k``` smallest value, called **k nearest neighbours (KNN)**. ```k``` can be 50, 100, 500, etc. You can choose any value for it.

* Step 5: Count and find in ```k``` labels which label has the largest frequency. That is the number this algorithm guess.

## 7/ Run code
Run file ```main.py```.

Run by this cmd: ```python main.py```

## 8/ Optimize Speed
Use **[C++](https://www.freecodecamp.org/news/the-c-plus-plus-programming-language/)** code to increase speed.
### 8.1/ Prepare
#### You must have **C++** compiler to compile **C++** code.

Get the ```lib.hpp``` and ```lib.cpp``` files.

Run these command (I use **[GNU-GCC](https://gcc.gnu.org/)**):
#### ``` g++ -c -fPIC lib.cpp -o lib.o ```
#### ``` g++ -shared lib.o -o lib.so ```

Or compile them by **[Visual Studio](https://visualstudio.microsoft.com/vs/)**.

You now have a ```lib.so``` file. Keep this file and ```main_optimze.py``` file in same directory.

#### If you don't want to edit the library or you don't have a compiler, use mine instead of building by yourself.
### 8.2/ Let's Rock!
Run ```main_optimize.py``` file instead of ```main.py``` file.

The only difference between these files is ```main.py``` runs ```guess()``` function in ```Python```, but in ```main_optimize.py```, the ```guess()``` function calls the ```guess_optimize()``` function written by **C++** in ```lib.so```.
