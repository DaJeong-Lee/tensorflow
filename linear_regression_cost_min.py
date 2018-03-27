import tensorflow as tf
import matplotlib.pyplot as plt

X = [1,2,3]
Y = [1,2,3]

#W = tf.placeholder(tf.float32)
W = tf.Variable(5.0)

#간단하게 계산하기위해 +b는 생략(값이 많이차이안나서 생략가능)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis-Y))

# get minizised cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
    #학습 횟수가 늘어날수록 W는 1.0에 수렴

'''
# 그래프 그리기
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1 #(-3~5 사이를 움직이는데 0.1씩 움직이겠음 -> learning rate )
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W}) # cost, W가 어떻게 변하는지
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the cost function
plt.plot(W_val, cost_val) # W를 x축, cost를 Y축으로 그래프 그림
plt.show()
'''