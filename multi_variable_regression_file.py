import numpy as np
import tensorflow as tf

xy = np.loadtxt('multi_variable_data.csv', delimiter=',', dtype=np.float32)

#numpy는 2차원 배열 slice기능 제공 (앞은 1차, 뒤는2차 array slice)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# x_data row는 한사람의 x1,x2,x3,y점수

X = tf.placeholder(tf.float32, shape=[None, 3]) #x_data는 갯수는 무한(None), 각 항목은 3개
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))

# get minizised cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step%100 == 0:
        print('{}, cost: {}, Predication: {}'.format(step, cost_val, hy_val))
        # hy_val이 y_data와 비슷


#Ask my score
print('Your score is {}'.format(sess.run(hypothesis, feed_dict={X: [[100, 71, 101]]})))