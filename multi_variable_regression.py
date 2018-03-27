import tensorflow as tf

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [[152.], [185.], [180.], [196.], [142.]]

x_data = []
for x1, x2, x3 in zip(x1_data, x2_data, x3_data):
    x_data.append([x1,x2,x3])

print(x_data) #[[73.0, 80.0, 75.0], [93.0, 88.0, 93.0], [89.0, 91.0, 90.0], [96.0, 98.0, 100.0], [73.0, 66.0, 70.0]]

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
    if step%10 == 0:
        print('{}, cost: {}, Predication: {}'.format(step, cost_val, hy_val))
        # hy_val이 y_data와 비슷



