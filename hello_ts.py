import tensorflow as tf

def hello():
    hello = tf.constant('Hello, Tensorflow')
    sess = tf.Session()
    print(sess.run(hello))

def calc():
    #node1+2 해서 node3 계산
    node1 = tf.constant(3.0)
    node2 = tf.constant(4.0)
    node3 = tf.add(node1, node2)

    sess = tf.Session()
    print(sess.run([node1, node2]))
    print(sess.run(node3))

def placeholder():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a+b

    sess = tf.Session()
    print(sess.run(adder_node, feed_dict={a:3, b:5}))
    print(sess.run(adder_node, feed_dict={a:[3,4], b:[4,5]}))

placeholder()
'''
1. bulid graph
2. session.run()
3. update variables in the graph
'''
