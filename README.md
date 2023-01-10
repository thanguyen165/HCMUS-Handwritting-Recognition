# HCMUS-Handwritting-Recognition

## 1/ Environment
### Python 3.7
Download at https://docs.conda.io/en/latest/miniconda.html
### Visual Studio Code (VS Code):
Download at https://code.visualstudio.com/Download

#### Install extension "Python" after installing VS Code.

## 2/ Prepare MNIST database
Download MNIST database at: http://yann.lecun.com/exdb/mnist/ and **DO NOT UNZIP FILES**

The MNIST database contains 60,000 images used to recognise input numbers called ```train```, and 10,000 images used to check if the algorithm is good or bad, called ```test```. Every image has its label, respective to the number written in the image.

## 3/ Organise project
4 zips of MNIST daatabase is in ```data``` subfolder

## 4/ Before we start...
Run ```test_MNIST.py``` file to make sure MNIST database is successfully installed and set up.

## 5/ How do we recognise the numbers?
Step 1: Vectorize all the images of ```train``` database and the ```input img```.

Step 2: Find the **distance** from ```input img``` and each img in ```train```.

Step 3: Sort all the distances in increasing order.

Step 4: Choose ```k``` smallest value, called **k nearest neighbour (KNN)**. ```k``` can be 50, 100, 500, etc. You can choose any value for it.

Step 5: Count and find in ```k``` labels which label has the largest frequency. That is the number this algorithm guess.
