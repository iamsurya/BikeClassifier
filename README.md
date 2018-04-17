# BikeClassifier
Image learning classifier based on the mnist example for Tensorflow.  
The code is adapted for Windows, which does not have case sensitive  
filenames.
Using the example code provided would lead to loading the same files  
twice, i.e. you would iterate twice instead of once.

## Usage
You need to have python, scipy and tensorflow installed.  

### Pre-processing
First prep the images. This converts the images into numpy arrays.  
We do this once and reuse the arrays to save time. (We want to spend  
time on training the classifier, not loading the files.)  

python prepdata.py  

### Training
Then train the classifier using:  

python train.py  

### Testing
Make sure there are some files in the test folder.  
Test the classifier using:  

python test.py