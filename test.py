# test.py

import os
import tensorflow as tf
import numpy as np
import cv2

# module-level variables ##############################################################################################
TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    #### Our images are 300 X 400 X 3
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    #### we have two classes, not 10
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        LL = tf.nn.softmax(y_conv)
        prediction = tf.argmax(LL, 1)

    return y_conv, keep_prob, prediction, LL


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#######################################################################################################################
def main():
    print("starting program . . .")

    # get a list of classifications from the labels file
    classifications = ['mountain bike', 'road bike'];

    # Recreate the model
    x = tf.placeholder(tf.float32, [None, 28*28])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob, prediction, LL = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    # if the test image directory listed above is not valid, show an error message and bail
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    # end if

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "trainedmodel/bikemodel.ckpt")
    
        # for each file in the test images directory . . .
        for fileName in os.listdir(TEST_IMAGES_DIR):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            # show the file name on std out
            print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            # attempt to open the image with OpenCV
            openCVImage = cv2.imread(imageFileWithPath)

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue
            # end if

            # Convert image to tensor
            im = cv2.imread(imageFileWithPath, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(im, (28, 28), interpolation = cv2.INTER_CUBIC) # Resize image to 28,28
            # !! 
            vector = res.reshape(-1, 28*28) # Flatten vector
            vector = vector / 255.0;             # Normalize

            b = LL.eval(feed_dict={x: vector,keep_prob: 1.0}, session=sess)

            #sortedPredictions = b.argsort()[-len(b):][::-1]
            #print(b);
            #print(np.argmax(b));
            #print("\n");
            prediction = np.argmax(b)
            scoreAsAPercent = np.max(b) * 100.0
            strClassification = classifications[prediction]
            print("the object appears to be a " + strClassification + ", " + "{0:.4f}".format(scoreAsAPercent) + "% confidence")
            writeResultOnImage(openCVImage, strClassification + ", " + "{0:.4}".format(scoreAsAPercent) + "% confidence")
            cv2.imshow(fileName, openCVImage);
            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()
    # end for
# end with

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

# end main

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
