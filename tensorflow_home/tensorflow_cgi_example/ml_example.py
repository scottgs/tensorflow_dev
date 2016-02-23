#import MNIST image data and tensorflow: approx 55,000 training images/labels, 10,000 test images
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#images are 28x28 -> reduces to 1D vector of dimension 784
#[None, 784] represents a tensor with 'shape' N x 784 where N indexes the image and 784 indexes the pixel
x = tf.placeholder(tf.float32, [None, 784])

# Weights variable corresponding to 784 x 10; i.e. for each digit, a weight is assigned to the likilihood of pixel i occuring
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# represents a 2-D Tensor with one dimension for images and one for the value it represents (the label)
y_ = tf.placeholder(tf.float32, [None, 10])

# Softmax model attempts to assign probabilities that an image is 1 of n objects.
# To compute, we add up the pixel intensities as a weighted sum. The sum is weighted
# such that it is positive if a given intensity indicates it is likely a pixel in the correct 'class' or object
# and it is negative otherwise. Think of the softmax computation is computing, for each label, the weighted liklihood at each pixel appearing
# in a '10'; this is the 'activation' function in a neural network
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Measure for evaluating our model,
# We aim to minimize this function.
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Setup the computational nodes to do back propagated gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Init all the variables as much as possible, lazily
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train stochastically
for i in range(100):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x : batch_x, y_: batch_y})

# Lets see how we did!
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
