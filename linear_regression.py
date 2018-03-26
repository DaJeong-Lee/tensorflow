import tensorflow as tf

def regression():
    # 1. building graph
    x_train = [1,2,3]
    y_train = [1,2,3]

    #Variable: tensorflow가 사용하는 variable
    #값을 모르니까 random하게 줌(일차원배열로): tf.random_normal([1])
    W = tf.Variable(tf.random_normal([1]), name = 'weight')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')

    #우리의 가설 Wx*b
    hypothesis = x_train * W +b

    #우리의 cost function (H(x) - y)^2 의 평균괎(reduce_mean())
    cost = tf.reduce_mean(tf.square(hypothesis-y_train))

    #Cost minizised -> GradientDescent(최소화 하기위한 여러가지 중 하나)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(cost)

    # 2. run/update graph and get results
    sess = tf.Session()
    # 위의 Variable을 실행하기전엔 global_bariables_initialzer() 해줘야함
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        #train 2000번 돌릴껀데 20번마다 출력
        if step % 20 == 0:
            print(step, sess.run(cost), sess.run(W), sess.run(b))

    #sess.run(train)은 그것과 연결된 cost, hypothesis, W, b를 실행시키는것임
    #위 데이터에서 정답은 W=1, b=0

    #결과값을 보면
    #처음에는 cost는 크고, W와 b는 랜덤값이었음
    #결과는 cost는 굉장히 작아지고, W는 1, b는 0에 수렴하고 있음.

    # placeholder로 trainning set을 줄 수있고, sess.run([a,b,c])에 배열을 넣어서 한번에 실행시킬 수있다.
    # placeholder를 쓰면 좋은게 X, Y값을 받아서 실행시킬 수있다.
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    hypothesis = X * W +b
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3], Y:[1,2,3]})
        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)

    # 3. Test our model
    print(sess.run(hypothesis, feed_dict={X:[5,6,7]}))

regression()
