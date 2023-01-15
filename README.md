# HCMUS-Handwritting-Recognition

## 1/ Author:
This is our project for Introduction to Information Technology.

Our team:

1. [Phuc, Song Dong Gia](https://github.com/songdongsongdongphuc/)

2. [Loi, Nguyen Minh](https://github.com/mf0212/)

3. [Thang, Nguyen Quang](https://github.com/thanguyen165/)

## 2/ Environment
### Python 3.7
Download at https://docs.conda.io/en/latest/miniconda.html
### Visual Studio Code (VS Code):
Download at https://code.visualstudio.com/Download/

#### Install extension "Python" after installing VS Code.

## 3/ Prepare MNIST database
Download MNIST database at: http://yann.lecun.com/exdb/mnist/ and **DO NOT UNZIP FILES**.

The MNIST database contains 60,000 images used to recognise input numbers called ```train```, and 10,000 images used to check if the algorithm is good or bad, called ```test```. Every image has its label, respective to the number written in the image.

## 4/ Organise project
4 zips of MNIST database is in ```data``` subfolder.

## 5/ Before we start...
Run ```test_MNIST.py``` file to make sure MNIST database is successfully installed and set up.

## 6/ How do we recognise the numbers?
Step 1: Vectorize all the images of ```train``` database and the ```input img```.

Step 2: Find the **distance** between ```input img``` and each img in ```train```.

Step 3: Sort all the distances in increasing order.

Step 4: Choose ```k``` smallest value, called **k nearest neighbours (KNN)**. ```k``` can be 50, 100, 500, etc. You can choose any value for it.

Step 5: Count and find in ```k``` labels which label has the largest frequency. That is the number this algorithm guess.

## 7/ Run code
Run file ```main.py``` and see what will happen.

## 8/ Optimize Speed
Use **[C++](https://www.freecodecamp.org/news/the-c-plus-plus-programming-language/)** code to increase speed.
### 7.1/ Prepare
#### You must have **C++** compiler to do this way.

Get the ```lib.hpp``` and ```lib.cpp``` files.

Run these command (I use **[GNU-GCC](https://gcc.gnu.org/)**):
#### ``` g++ -c -fPIC lib.cpp -o lib.o ```
#### ``` g++ -shared lib.o -o lib.so ```

Or compile them by **[Visual Studio](https://visualstudio.microsoft.com/vs/)**
### 7.2/ Let's Rock!
Run ```main_C.py``` file instead of ```main.py``` file.

The only difference of these files is ```main_C.py``` runs ```guess()``` function in **C++** (which in ```lib.cpp``` file), but ```main.py``` runs that function in ```Python```.
