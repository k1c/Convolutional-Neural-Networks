"""
COMP 4107 Assignment #4

Carolyne Pelletier: 101054962
Akhil Dalal: 100855466

2x2 max pool filter
1 0.445
2 0.559
3 0.633
4 0.63
5 0.7
6 0.687
7 0.711
8 0.715
9 0.721
10 0.705
11 0.719
12 0.77
13 0.737
14 0.715
15 0.734
Elapsed 36.05 minutes.

3x3 max pool filter
1 0.41
2 0.547
3 0.581
4 0.665
5 0.683
6 0.699
7 0.69
8 0.694
9 0.701
10 0.71
11 0.663
12 0.707
13 0.719
14 0.725
15 0.708
Elapsed 22.48 minutes.

"""
import tensorflow as tf
import numpy as np
import time

batch_size = 128
test_size = 1000

def init_weights(shape,stddev_layer,name):
    return tf.Variable(tf.random_normal(shape, stddev=stddev_layer),name=name)

#ksize: A 1-D int Tensor of 4 elements. The size of the window for each dimension of the input tensor.
#strides: A 1-D int Tensor of 4 elements. The stride of the sliding window for each dimension of the input tensor.
#padding: 'SAME' adds the zero padding
def model(X, filter_conv1, filter_conv2, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act):
    #input size: 32x32x3
    #filter: 32 3x3x3
    #conv1 size: 32x32x32
    conv1 = act(tf.nn.conv2d(X, filter_conv1, strides=[1, 1, 1, 1], padding='SAME', name='conv1'))

    #input size: 32x32x32
    #pool1 size: 16x16x32
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1') #is this adding padding? 'VALID'
    dropout1 = tf.nn.dropout(pool1, p_keep_conv, name='dropout1')

    #input size: 16x16x32
    #filter: 64 3X3X32
    #conv2 size: 16x16x64
    conv2 = act(tf.nn.conv2d(dropout1, filter_conv2, strides=[1, 1, 1, 1], padding='SAME', name='conv2'))

    #input size: 16x16x64
    #pool2 size: 8x8x64
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    dropout2 = tf.nn.dropout(pool2, p_keep_conv, name='dropout2')

    #flatten: 1x(8*8*64)
    flatten = tf.reshape(dropout2, [-1, weight_fc.get_shape().as_list()[0]])

    #matrix_mult: 1x4096 * 4096x512
    fc1 = act(tf.matmul(flatten, weight_fc, name='fc1'))
    dropout3 = tf.nn.dropout(fc1, p_keep_hidden, name='dropout3')
    
    #matrix_mult: 1x512 * 512x512
    fc2 = act(tf.matmul(dropout3, weight_fc2, name='fc2'))
    dropout4 = tf.nn.dropout(fc2, p_keep_hidden, name='dropout4')
    
    #matrix_mult: 1x512 * 512x10
    pyx = tf.matmul(dropout4, weight_output, name='pyx')

    #returns: 1x10
    return pyx

# 32x32x3 input img placeholder
X = tf.placeholder("float", [None, 32, 32, 3], name='image')

# 1X10 output placeholder
Y = tf.placeholder("float", [None, 10], name='label')

#calculate standard deviations for each filter with weights
#filter_img connects to conv layer which connects to pool_layer (no depth since 1 filter produces 1 activation map)
stddev_filter_conv1 = np.sqrt(2) * np.sqrt(2/(3*3*3 + 16*16))
stddev_filter_conv2 = np.sqrt(2) * np.sqrt(2/(3*3*32 + 8*8))
stddev_weight_fc = np.sqrt(2) * np.sqrt(2/((8*8*64) + 512))
stddev_weight_fc2 = np.sqrt(2) * np.sqrt(2/(512 + 512))

# 32 filters 3x3x3
filter_conv1 = init_weights([3, 3, 3, 32], stddev_filter_conv1,'filter_conv1')

# 64 filters 3x3x32
filter_conv2 = init_weights([3, 3, 32, 64], stddev_filter_conv2,'filter_conv2')

# (8*8*64)x512
weight_fc = init_weights([8*8*64, 512], stddev_weight_fc,'weight_fc')

# 512x512
weight_fc2 = init_weights([512, 512], stddev_weight_fc2, 'weight_fc2')

# 512x10
weight_output = init_weights([512, 10],0.01,'weight_output')

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

# act specifies the activation function to be used in the network
act = tf.nn.relu
#act = tf.nn.leaky_relu
py_x = model(X, filter_conv1, filter_conv2, weight_fc, weight_fc2, weight_output, p_keep_conv, p_keep_hidden, act)

#logits and labels must be same size: logits_size=[128,10] labels_size=[128,10]   (128 represents the batch size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter("cnn/cnn-2",sess.graph)
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

