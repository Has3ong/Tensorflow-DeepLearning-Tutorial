import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# layer
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))
Layer1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
Layer1 = tf.nn.relu(Layer1)
Layer1 = tf.nn.max_pool(Layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
Layer2 = tf.nn.conv2d(Layer1, W2, strides=[1, 1, 1, 1], padding='SAME')
Layer2 = tf.nn.relu(Layer2)
Layer2 = tf.nn.max_pool(Layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Layer2 = tf.reshape(Layer2, [-1, 7 * 7 * 64])

W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(Layer2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# init
sess = tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs = 15
batch_size = 100

# train
print('Learning started. It takes sometimes.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print('Learning Finished!')

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))