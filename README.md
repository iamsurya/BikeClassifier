# BikeClassifier
Image learning classifier using the [mnist-deep](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py) (1) example for Tensorflow.  
Some code from [MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through
](https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through) (2)  
was used for parsing file lists and for test.py.
The code from (2) is adapted for Windows,  
which does not have case sensitive filenames. The 
original code would load the files twice  
because looking for .jpeg and .JPEG would 
both return a file. Using the example code provided  
would lead to loading the same files twice, i.e. you would iterate twice instead of once.  

Example of mountain_bike_0.jpg  
![](https://github.com/iamsurya/BikeClassifier/raw/master/example_mountain_bike_0.jpg)

Example of road_bike_10.jpg  
![](https://github.com/iamsurya/BikeClassifier/raw/master/example_road_bike_10.jpg)

## Usage
You need to have python, scipy (preferably anaconda) and tensorflow installed.  
Then run the following commands to create a dataset, train and then test.

### Clone this git repository

`git clone https://github.com/iamsurya/BikeClassifier.git`

### Pre-processing
First prep the images. This converts the images into numpy arrays.  
We do this once and reuse the arrays to save time. (If we had larger images,  
We would want to spend time on training the classifier, not loading  
the files.)  

`python prepdata.py`  

### Training
Then train the classifier using:  

`python train.py`  

### Testing
Make sure there are some files in the test folder.  
Test the classifier using:  

`python test.py`

## Notes
Some notes on the dataset are in [VisBikes.ipynb](https://github.com/iamsurya/BikeClassifier/blob/master/VisBikes.ipynb) where I try to visualize and look at the input dataset.
The file also covers notes on features in the images that might help with classification if using a traditional computer vision approach.
The MNIST example provided accepts 28X28 grayscale images as input. I decided to reduce the input images to this resolution and feed it to the network. The MNIST dataset has 60,000 images, we have ~200 so we do expect to train faster, but downsampling and reducing the images to this resolution means we are throwing away a lot of information available in a 1000X1000 colored image. However, we do reduce the images to important features like edges by resizing them, which are easier to learn for an NN (see [VisBikes.ipynb](https://github.com/iamsurya/BikeClassifier/blob/master/VisBikes.ipynb) for thoughts).

This had the following consequences:
* The number of iterations required for training were reduced drastically. The original example using inception used 2000 iteration. In comparison, 100 iterations show 100% accuracy on the validation and test set.
* Time required to train is reduced.  

On the test set, confidence is shown to be 100% or close to 100%. Images that contained no colors (black and white) show 100% confidence. The presence of color seems to reduce this to 99%, which is a result of the colors affecting brightness levels when converted to grayscale.  

Amusingly, these results are much better than those shown in the [MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through
](https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through) example which uses Google's inception model on full resolution colored images, showing how pre-processing and feature reduction can improve training and classification vastly, and we should not fully depend on a deep network.  

Given that we needed only 100 iterations to classify well, we could deal with this classification problem with traditional CV methods like edge detection, template matching, PCA and / or SIFT. An example of the average image for the two classes shows this as well, because the bikes can be reduced to just their edges and frames. See [VisBikes.ipynb](https://github.com/iamsurya/BikeClassifier/blob/master/VisBikes.ipynb) for more images.  

![](https://github.com/iamsurya/BikeClassifier/raw/master/averageimg.jpg)

## Model

The main graph for the network used in the [mnist-deep example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py) from the [Tensorboard Screenshot](https://github.com/iamsurya/BikeClassifier/raw/master/TensorboardScreenshot.PNG) shows the following network:  
### \[(Conv -> Pool) X 2\] -> FC1 -> Dropout -> FC2 -> Y
![MainGraph](https://github.com/iamsurya/BikeClassifier/raw/master/main-graph.png)