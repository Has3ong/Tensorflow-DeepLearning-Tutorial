import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x))


# y = ax + b + noise_levels
b = 5
a = 2
y = (a * x) + b + noise


my_data = pd.concat([pd.DataFrame(data=x, columns=['X Data']), pd.DataFrame(data=y, columns=['Y'])], axis=1)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')


batch_size = 8
a = tf.Variable(2.0)
b = tf.Variable(5.0)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])
y_model = a * xph + b

error = tf.reduce_sum(tf.square(yph-y_model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 1000

    for i in range(batches):
        rand_ind = np.random.randint(len(x), size=batch_size)

        feed = {xph: x[rand_ind], yph: y[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([a, b])

    y_hat = x * model_m + model_b
    my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
    plt.plot(x, y_hat, 'r')
    plt.show()