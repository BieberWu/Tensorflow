# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:43:11 2018

@author: hp
"""

Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 6.4.0 -- An enhanced Interactive Python.

import tensorflow as tf
C:\Users\hp\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters

import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0, 0.05, x_data.shape)



y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32, [None, 1])

ys=tf.placeholder(tf.float32, [None, 1])

def add_layer(inputs, in_size, out_size, activation_function=None)
  File "<ipython-input-7-c6f5b66fb7ab>", line 1
    def add_layer(inputs, in_size, out_size, activation_function=None)
                                                                      ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    

Wx_plus_b=tf.matmul(inputs, Weights)+biases
Traceback (most recent call last):

  File "<ipython-input-9-96e652a2fdb3>", line 1, in <module>
    Wx_plus_b=tf.matmul(inputs, Weights)+biases

NameError: name 'inputs' is not defined




def add_layer(inputs, in_size, out_size, activation_function=None):
   Weights=tf.Variable(tf.random_normal([in_size, out_size]))
   biases=tf.Variable(tf.zeros([1, out_size])+0.1)
   Wx_plus_b=tf.matmul(inputs, Weights)+biases
   if activation_function is None:
       outputs=Wx_plus_b
   else:
       outputs=activation_function(Wx_plus_b)
   return outputs



l1=add_layer(xs, 1, 10, activation_function=tf.nn.relu)


prediction=add_layer(l1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

optimizer=tf.train.GradientDescentOptimizer(0.1)

train_step=optimizer.minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()

sess.run(int)
Traceback (most recent call last):

  File "<ipython-input-18-c45aa5bfce9e>", line 1, in <module>
    sess.run(int)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 887, in run
    run_metadata_ptr)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1095, in _run
    self._graph, fetches, feed_dict_tensor, feed_handles=feed_handles)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 429, in __init__
    self._fetch_mapper = _FetchMapper.for_fetch(fetches)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 255, in for_fetch
    return _ElementFetchMapper(fetches, contraction_fn)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 288, in __init__
    (fetch, type(fetch), str(e)))

TypeError: Fetch argument <class 'int'> has invalid type <class 'type'>, must be a string or Tensor. (Can not convert a type into a Tensor or Operation.)




sess=tf.Session()
sess.run(int)


Traceback (most recent call last):

  File "<ipython-input-19-b0a58c8431f2>", line 2, in <module>
    sess.run(int)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 887, in run
    run_metadata_ptr)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1095, in _run
    self._graph, fetches, feed_dict_tensor, feed_handles=feed_handles)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 429, in __init__
    self._fetch_mapper = _FetchMapper.for_fetch(fetches)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 255, in for_fetch
    return _ElementFetchMapper(fetches, contraction_fn)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 288, in __init__
    (fetch, type(fetch), str(e)))

TypeError: Fetch argument <class 'int'> has invalid type <class 'type'>, must be a string or Tensor. (Can not convert a type into a Tensor or Operation.)




for in in range(1000):
  File "<ipython-input-20-1126c8c51ae7>", line 1
    for in in range(1000):
         ^
SyntaxError: invalid syntax




for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
    print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
  File "<ipython-input-21-103e34fefee1>", line 4
    print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
        ^
IndentationError: expected an indented block




for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
        print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
        
Traceback (most recent call last):

  File "<ipython-input-22-053b61548433>", line 2, in <module>
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 887, in run
    run_metadata_ptr)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1110, in _run
    feed_dict_tensor, options, run_metadata)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1286, in _do_run
    run_metadata)

  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1308, in _do_call
    raise type(e)(node_def, op, message)

FailedPreconditionError: Attempting to use uninitialized value Variable
	 [[{{node Variable/read}} = Identity[T=DT_FLOAT, _class=["loc:@GradientDescent/update_Variable/ApplyGradientDescent"], _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable)]]

Caused by op 'Variable/read', defined at:
  File "C:\Users\hp\Anaconda3\lib\site-packages\spyder\utils\ipython\start_kernel.py", line 269, in <module>
    main()
  File "C:\Users\hp\Anaconda3\lib\site-packages\spyder\utils\ipython\start_kernel.py", line 265, in main
    kernel.start()
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\kernelapp.py", line 486, in start
    self.io_loop.start()
  File "C:\Users\hp\Anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 127, in start
    self.asyncio_loop.run_forever()
  File "C:\Users\hp\Anaconda3\lib\asyncio\base_events.py", line 422, in run_forever
    self._run_once()
  File "C:\Users\hp\Anaconda3\lib\asyncio\base_events.py", line 1432, in _run_once
    handle._run()
  File "C:\Users\hp\Anaconda3\lib\asyncio\events.py", line 145, in _run
    self._callback(*self._args)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 117, in _handle_events
    handler_func(fileobj, events)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tornado\stack_context.py", line 276, in null_wrapper
    return fn(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\zmq\eventloop\zmqstream.py", line 450, in _handle_events
    self._handle_recv()
  File "C:\Users\hp\Anaconda3\lib\site-packages\zmq\eventloop\zmqstream.py", line 480, in _handle_recv
    self._run_callback(callback, msg)
  File "C:\Users\hp\Anaconda3\lib\site-packages\zmq\eventloop\zmqstream.py", line 432, in _run_callback
    callback(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tornado\stack_context.py", line 276, in null_wrapper
    return fn(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 283, in dispatcher
    return self.dispatch_shell(stream, msg)
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 233, in dispatch_shell
    handler(stream, idents, msg)
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\kernelbase.py", line 399, in execute_request
    user_expressions, allow_stdin)
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\ipkernel.py", line 208, in do_execute
    res = shell.run_cell(code, store_history=store_history, silent=silent)
  File "C:\Users\hp\Anaconda3\lib\site-packages\ipykernel\zmqshell.py", line 537, in run_cell
    return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2662, in run_cell
    raw_cell, store_history, silent, shell_futures)
  File "C:\Users\hp\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2785, in _run_cell
    interactivity=interactivity, compiler=compiler, result=result)
  File "C:\Users\hp\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2903, in run_ast_nodes
    if self.run_code(code, result):
  File "C:\Users\hp\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py", line 2963, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-11-54756e3e9fc7>", line 1, in <module>
    l1=add_layer(xs, 1, 10, activation_function=tf.nn.relu)
  File "<ipython-input-10-741b74f5b86f>", line 2, in add_layer
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 145, in __call__
    return cls._variable_call(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 141, in _variable_call
    aggregation=aggregation)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 120, in <lambda>
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variable_scope.py", line 2441, in default_variable_creator
    expected_shape=expected_shape, import_scope=import_scope)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 147, in __call__
    return super(VariableMetaclass, cls).__call__(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 1104, in __init__
    constraint=constraint)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\variables.py", line 1266, in _init_from_args
    self._snapshot = array_ops.identity(self._variable, name="read")
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\array_ops.py", line 81, in identity
    return gen_array_ops.identity(input, name=name)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\ops\gen_array_ops.py", line 3353, in identity
    "Identity", input=input, name=name)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\util\deprecation.py", line 488, in new_func
    return func(*args, **kwargs)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 3272, in create_op
    op_def=op_def)
  File "C:\Users\hp\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 1768, in __init__
    self._traceback = tf_stack.extract_stack()

FailedPreconditionError (see above for traceback): Attempting to use uninitialized value Variable
	 [[{{node Variable/read}} = Identity[T=DT_FLOAT, _class=["loc:@GradientDescent/update_Variable/ApplyGradientDescent"], _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable)]]





def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    if activation_function=None:
  File "<ipython-input-23-0f5b667871c4>", line 6
    if activation_function=None:
                          ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    if activation_function=None :
  File "<ipython-input-24-9ac399ee2891>", line 6
    if activation_function=None :
                          ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    if activation_function is None
  File "<ipython-input-25-fb151747f04e>", line 6
    if activation_function is None
                                  ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    if activation_function is None :
        outputs=Wx_plus_b
    else
  File "<ipython-input-26-82dc46a24925>", line 8
    else
        ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size, out_size]))
    biases=tf.Variable(tf.zeros([1, out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs, Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    if activation_function is None :
        outputs=Wx_plus_b
    else :
        outputs=activation_function(Wx_plus_b)
    return outputs



import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer')
  File "<ipython-input-29-4a951d64e098>", line 2
    with tf.name_scope('layer')
                               ^
SyntaxError: invalid syntax




def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights=tf.variable(tf.random_normal([in_size, out_size]), name='w')
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(inputs, weights), biases)
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b,)
        return outputs
    

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')
    

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)



with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer())
