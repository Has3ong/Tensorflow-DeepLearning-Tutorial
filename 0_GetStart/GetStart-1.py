import tensorflow as tf

sess = tf.InteractiveSession()

tensor = tf.random_uniform((4, 4), 0, 1)
var = tf.Variable(initial_value=tensor)

# <tf.Variable 'Variable:0' shape=(4, 4) dtype=float32_ref>
print(var)

init = tf.global_variables_initializer()
init.run()


print(var.eval())
"""
[[0.12531924 0.29500675 0.04305232 0.24124885]
 [0.47991002 0.31639528 0.20714486 0.39324558]
 [0.24083173 0.72731173 0.46757686 0.07476521]
 [0.17398047 0.04015446 0.1643374  0.7555349 ]]
"""

print(sess.run(var))


ph = tf.placeholder(tf.float32, shape=(5, 5))

# Tensor("Placeholder:0", shape=(5, 5), dtype=float32)
print(ph)
