try:
	import numpy as np
	print('Numpy succesfully imported\n')
except:
	print('Error : Numpy not installed properly\n')
try:
	import pyaudio
	print('Pyaudio succesfully imported\n')
except:
	print('Error : pyaudio not installed properly\n')

print('Importing tensorflow \nThis might take some time\n')
try:
	import tensorflow as tf
	hello = tf.constant('Hello, TensorFlow!')
	sess = tf.Session()
	if sess.run(hello) == b'Hello, TensorFlow!':	
		print('Succesful imported string element')
	a = tf.constant(10)
	b = tf.constant(32)
	if sess.run(a + b) == 42:
		print('Succesful imported math element')
	sess.close()
except:
	print('Error : tensorflow not installed properly')
