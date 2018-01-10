"""
COMP 4107 Assignment #4

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

r = ReLU
d = Dropout
s = Softmax
In -> {Conv (r) -> Conv (r) - > Pool (d)}*3 -> Flatten -> FC (r) -> FC (r) -> Out (s)

2x2 max pool filter
1 0.331
2 0.507
3 0.611
4 0.621
5 0.666
6 0.7
7 0.713
8 0.688
9 0.736
10 0.767
11 0.733
12 0.746
13 0.778
14 0.768
15 0.765
Elapsed 73.94 minutes.

3x3 max pool filter
1 0.326
2 0.549
3 0.639
4 0.662
5 0.68
6 0.723
7 0.686
8 0.752
9 0.73
10 0.754
11 0.752
12 0.772
13 0.677
14 0.754
15 0.771
Elapsed 120.12 minutes (this ran on my laptop, hence more time taken).

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
def model(X, filter_conv1, filter_conv2, filter_conv3, filter_conv4, filter_conv5, filter_conv6, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act):
    # --- CCP 1
    #input size: ?x32x32x3
    #filter: 32 3x3x3
    #conv1 size: ?x32x32x32
    conv1 = act(tf.nn.conv2d(X, filter_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv1'))
    
    #input size: ?x32x32x32
    #filter: 32 3x3x32
    #conv2 size: ?x32x32x32
    conv2 = act(tf.nn.conv2d(conv1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME', name='conv2'))

    #input size: ?x32x32x32
    #pool1 size: ?x16x16x32
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    dropout1 = tf.nn.dropout(pool1, p_keep_conv, name='dropout1')

    # --- CCP 2
    #input size: ?x16x16x32
    #filter: 64 3x3x32
    #conv3 size: ?x16x16x64
    conv3 = act(tf.nn.conv2d(dropout1, filter_conv3, strides=[1, 1, 1, 1], padding='SAME', name='conv3'))
    
    #input size: ?x16x16x64
    #filter: 64 3x3x64
    #conv2 size: ?x16x16x64
    conv4 = act(tf.nn.conv2d(conv3, filter_conv4, strides=[1, 1, 1, 1], padding='SAME', name='conv4'))

    #input size: ?x16x16x64
    #pool2 size: ?x8x8x64
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    dropout2 = tf.nn.dropout(pool2, p_keep_conv, name='dropout2')
    
    # --- CCP 3
    #input size: ?x8x8x64
    #filter: 128 3x3x64
    #conv3 size: ?x8x8x128
    conv5 = act(tf.nn.conv2d(dropout2, filter_conv5, strides=[1, 1, 1, 1], padding='SAME', name='conv5'))
    
    #input size: ?x8x8x128
    #filter: 128 3x3x64
    #conv2 size: ?x8x8x128
    conv6 = act(tf.nn.conv2d(conv5, filter_conv6, strides=[1, 1, 1, 1], padding='SAME', name='conv6'))

    #input size: ?x8x8x128
    #pool2 size: ?x4x4x128
    pool3 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    dropout3 = tf.nn.dropout(pool3, p_keep_conv, name='dropout3')
    
    #flatten: 1x(4*4*128)
    flatten = tf.reshape(dropout3, [-1, weight_fc.get_shape().as_list()[0]])

    #matrix_mult: 1x2048 * 2048x512
    fc1 = act(tf.matmul(flatten, weight_fc, name='fc1'))
    dropout4 = tf.nn.dropout(fc1, p_keep_hidden, name='dropout4')

    #matrix_mult: 1x512 * 512x512
    fc2 = act(tf.matmul(dropout4, weight_fc2, name='fc2'))
    dropout5 = tf.nn.dropout(fc2, p_keep_hidden, name='dropout5')

    #matrix_mult: 1x512 * 512x10
    pyx = tf.matmul(dropout5, weight_output, name='pyx')

    #returns: 1x10
    return pyx

# 32x32x3 input img placeholder
X = tf.placeholder("float", [None, 32, 32, 3], name='image')

# 1X10 output placeholder
Y = tf.placeholder("float", [None, 10], name='label')

#calculate standard deviations for each filter with weights
stddev_filter_conv1 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*3.0) + (32.0*32.0)))
stddev_filter_conv2 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*32.0) + (16.0*16.0)))
stddev_filter_conv3 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*32.0) + (16.0*16.0)))
stddev_filter_conv4 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*64.0) + (8.0*8.0)))
stddev_filter_conv5 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*64.0) + (8.0*8.0)))
stddev_filter_conv6 = np.sqrt(2) * np.sqrt(2.0/((3.0*3.0*128.0) + (4.0*4.0)))
stddev_weight_fc = np.sqrt(2) * np.sqrt(2.0/((4.0*4.0*128.0) + 512.0))
stddev_weight_fc2 = np.sqrt(2) * np.sqrt(2.0/(512.0 + 512.0))

# 32 filters 3x3x3
filter_conv1 = init_weights([3, 3, 3, 32], stddev_filter_conv1,'filter_conv1')

# 32 filters 3x3x3
filter_conv2 = init_weights([3, 3, 32, 32], stddev_filter_conv2,'filter_conv2')

# 64 filters 3x3x32
filter_conv3 = init_weights([3, 3, 32, 64], stddev_filter_conv3,'filter_conv3')

# 64 filters 3x3x64
filter_conv4 = init_weights([3, 3, 64, 64], stddev_filter_conv4,'filter_conv4')

# 128 filters 3x3x64
filter_conv5 = init_weights([3, 3, 64, 128], stddev_filter_conv5,'filter_conv5')

# 128 filters 3x3x128
filter_conv6 = init_weights([3, 3, 128, 128], stddev_filter_conv6,'filter_conv6')

# (4*4*128)x512
weight_fc = init_weights([4*4*128, 512], stddev_weight_fc,'weight_fc')

# 512x512
weight_fc2 = init_weights([512, 512], stddev_weight_fc2,'weight_fc2')

# 512x10
weight_output = init_weights([512, 10], 0.01,'weight_output')

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# act specifies the activation function to be used in the network
act = tf.nn.relu
#act = tf.nn.leaky_relu
py_x = model(X, filter_conv1, filter_conv2, filter_conv3, filter_conv4, filter_conv5, filter_conv6, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act)


#logits and labels must be same size: logits_size=[128,10] labels_size=[128,10]   (128 represents the batch size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("cnn/cnn-4",sess.graph)

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

