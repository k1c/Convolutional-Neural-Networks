"""
COMP 4107 Assignment #4

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

r = ReLU
d = Dropout
s = Softmax
In -> Conv (r) -> Pool (d) -> Flatten -> FC (r) -> FC (r) -> Out (s)

2x2 max pool filter
1 0.403
2 0.531
3 0.605
4 0.589
5 0.644
6 0.649
7 0.626
8 0.634
9 0.637
10 0.645
11 0.662
12 0.684
13 0.673
14 0.664
15 0.67
Elapsed 14.39 minutes.

3x3 max pool filter
1 0.412
2 0.554
3 0.608
4 0.622
5 0.638
6 0.615
7 0.639
8 0.652
9 0.679
10 0.646
11 0.664
12 0.675
13 0.68
14 0.687
15 0.675
Elapsed 15.00 minutes.

"""
import tensorflow as tf
import numpy as np
import time

batch_size = 128
test_size = 1000


def init_weights(shape, stddev_layer, name):
    return tf.Variable(tf.random_normal(shape, stddev=stddev_layer),name=name)

#ksize: A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
#strides: A 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension of the input tensor.
#padding: 'SAME' adds the zero padding
def model(X, filter_conv1, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act):
    #input size: ?x32x32x3
    #filter: 32 3x3x3
    #conv1 size: ?x32x32x32
    conv1 = act(tf.nn.conv2d(X, filter_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv1'))

    #input size: ?x32x32x32
    #pool1 size: ?x16x16x32
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    dropout1 = tf.nn.dropout(pool1, p_keep_conv, name='dropout1')

    #flatten: 1x(16*16*32)
    flatten = tf.reshape(dropout1, [-1, weight_fc.get_shape().as_list()[0]])

    #matrix_mult: 1x8192 * 8192x512
    fc1 = act(tf.matmul(flatten, weight_fc, name='fc1'))
    dropout2 = tf.nn.dropout(fc1, p_keep_hidden, name='dropout2')

    #matrix_mult: 1x512 * 512x512
    fc2 = act(tf.matmul(dropout2, weight_fc2, name='fc2'))
    dropout3 = tf.nn.dropout(fc2, p_keep_hidden, name='dropout3')

    #matrix_mult: 1x512 * 512x10
    pyx = tf.matmul(dropout3, weight_output, name='pyx')

    #returns: 1x10
    return pyx

# 32x32x3 input img placeholder
X = tf.placeholder("float", [None, 32, 32, 3], name='image')

# 1X10 output placeholder
Y = tf.placeholder("float", [None, 10], name='label')

#calculate standard deviations for each filter with weights
stddev_filter_conv1 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*3.0) + (16.0*16.0)))
stddev_weight_fc = np.sqrt(2) * np.sqrt(2.0/((16.0*16.0*32.0) + 512.0))
stddev_weight_fc2 = np.sqrt(2) * np.sqrt(2.0/(512.0 + 512.0))

# 32 filters 3x3x3
filter_conv1 = init_weights([3, 3, 3, 32], stddev_filter_conv1,'filter_conv1')

# (16*16*32)x512
weight_fc = init_weights([16*16*32, 512], stddev_weight_fc,'weight_fc')

# 512x512
weight_fc2 = init_weights([512, 512], stddev_weight_fc2,'weight_fc2')

# 512x10
weight_output = init_weights([512, 10], 0.01,'weight_output')

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# act specifies the activation function to be used in the network
act = tf.nn.relu
#act = tf.nn.leaky_relu
py_x = model(X, filter_conv1, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act)


#logits and labels must be same size: logits_size=[128,10] labels_size=[128,10]   (128 represents the batch size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("cnn/cnn-1",sess.graph)

    # load data - one_hot Y vectors
    (trX, trY), (teX, teY) = tf.keras.datasets.cifar10.load_data()
    trX = trX/255.0
    teX = teX/255.0
    trY = tf.one_hot(trY[:, 0], 10).eval()
    teY = tf.one_hot(teY[:, 0], 10).eval()
    print("Training with:")
    print("-- X: ", trX.shape)
    print("-- Y: ", trY.shape)
    print("Testing with:")
    print("-- X: ", teX.shape)
    print("-- Y: ", teY.shape)
    
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    time_start = time.perf_counter()
    for i in range(15):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})


        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        print(i+1, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

    elapsed = (time.perf_counter() - time_start)/60.0
    print('Elapsed %.2f minutes.' % elapsed)
    writer.close()

