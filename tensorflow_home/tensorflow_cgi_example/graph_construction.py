import tensorflow as tf

# Basic Matrix Ops
A = tf.constant([[3,3]])
B = tf.constant([[2],[2]])
C = tf.matmul(A,B)

# Launch Computation
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run([C])
    print (result)
