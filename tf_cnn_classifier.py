#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
"""
# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

#import load_data
import _pickle as pickle
import gzip

import network

in_dir='data'
BATCH_SIZE = 5

height, width, color =  64, 64, 3
neural_network = network.conv_network_1
	
shape = height, width, color

import os.path
if os.path.exists('.notebook'):
	DISPLAY_INTERVAL, NUM_ITERS = 1, 5
else:
	DISPLAY_INTERVAL, NUM_ITERS = 10, 5000

f = gzip.open('dump.gz', 'rb')
data = pickle.load(f)

train = data['train']
valid = data['valid']
test  = data['test']
print('train size:', train['size'])
print('valid size:', valid['size'])
print('test size:', test['size'])
im0 = train['images'][0]
print('Data was loaded.')
print(im0.shape)
#sys.exit()

#train['images'] = [np.transpose(t) for t in train['images']]
num_train_batches = train['size'] // BATCH_SIZE
num_valid_batches = valid['size'] // BATCH_SIZE
num_test_batches = test['size'] // BATCH_SIZE
print('num_train_batches:', num_train_batches)
print('num_valid_batches:', num_valid_batches)
print('num_test_batches:', num_test_batches)
SAMPLE_SIZE = train['size']
max_valid_accuracy = 0

print('Example of data. Item 0:')
print('x:', train['images'][0])
print('y:', train['labels'][0])

# Create a new graph
graph = tf.Graph() # no necessiry

with graph.as_default():

	number_classes = 2
	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [None, height, width, color]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [None, number_classes])   # Placeholder for labels.
	#x_image = tf.reshape(x, [-1, height, width, color])
	x_image = tf.reshape(x, [-1, 64, 64, 3])
	logits = neural_network(x_image, output_size=number_classes)
	print('logits =', logits)

	# 2. Add nodes that represent the optimization algorithm.
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)

	optimizer= tf.train.AdagradOptimizer(0.005)
	#optimizer= tf.train.AdamOptimizer(0.005)
	#train_op = tf.train.GradientDescentOptimizer(0.01)
	train_op = optimizer.minimize(loss)
		
	# for classification:
	#loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
	#train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
	
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		for iteration in range(NUM_ITERS):			  # Train iteratively for NUM_iterationS.		 

			if iteration % (50*DISPLAY_INTERVAL) == 0:
				# model validation on valid data

				#output_values = logits.eval(feed_dict = {x:valid['images'][:2]})
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[0], valid['labels'][0]))
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[1], valid['labels'][1]))
				#print('valid: {0:.2f} - {1:.2f}'.format(output_values[2][0]*360, valid['labels'][2][0]*360))
				
				"""
				for i in range(num_valid_batches):
					feed_dict = {x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}
					#print(feed_dict)
					output_values = logits.eval(feed_dict=feed_dict)
					print(i, output_values)
					#print(output_values.shape)
				"""
				pass

			if iteration % (5*DISPLAY_INTERVAL) == 0:

				train_loss = np.mean( [loss.eval( \
					feed_dict={x:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_train_batches)])
				valid_loss = np.mean([ loss.eval( \
					feed_dict={x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_valid_batches)])

				train_accuracy = np.mean( [accuracy.eval( \
					feed_dict={x:train['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:train['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_train_batches)])
				valid_accuracy = np.mean([ accuracy.eval( \
					feed_dict={x:valid['images'][i*BATCH_SIZE:(i+1)*BATCH_SIZE], \
					y:valid['labels'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]}) \
					for i in range(0,num_valid_batches)])

				if valid_accuracy > max_valid_accuracy:
					max_valid_accuracy = valid_accuracy

				#min_in_grad = math.sqrt(min_valid_accuracy) * 360.0
				#min_in_grad = min_valid_accuracy * 360.0
				
				print('iter {0:3}: train={1:0.3f} [{2:0.2f}%], valid={3:0.3f} [{4:0.2f}%] (max={5:0.2f}%)'.\
					format(iteration, train_loss, 100*train_accuracy, valid_loss, 100*valid_accuracy, max_valid_accuracy))

				"""
				#train_accuracy = loss.eval(feed_dict = {x:train['images'][0:BATCH_SIZE], y:train['labels'][0:BATCH_SIZE]})
				#valid_accuracy = loss.eval(feed_dict = {x:valid['images'][0:BATCH_SIZE], y:valid['labels'][0:BATCH_SIZE]})
				"""
			
			a1 = iteration*BATCH_SIZE % train['size']
			a2 = (iteration + 1)*BATCH_SIZE % train['size']
			x_data = train['images'][a1:a2]
			y_data = train['labels'][a1:a2]
			if len(x_data) <= 0: continue
			sess.run(train_op, {x: x_data, y: y_data})  # Perform one training iteration.		
			#print(a1, a2, y_data)			

		# Save the comp. graph
		"""
		print('Save the comp. graph')
		x_data, y_data =  valid['images'], valid['labels'] #mnist.train.next_batch(BATCH_SIZE)		
		writer = tf.summary.FileWriter("output", sess.graph)
		print(sess.run(train_op, {x: x_data, y: y_data}))
		writer.close()  
		"""

		# Test of model
		#HERE SOME ERROR ON GPU OCCURS
		test_accuracy = np.mean([ loss.eval( \
			feed_dict={x:test['images'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]}) \
			for i in range(num_test_batches) ])
		
		for i in range(num_test_batches):
			feed_dict = {x:test['images'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]}
			labels = test['labels'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]
			filenames = test['filenames'][i*BATCH_SIZE : (i+1)*BATCH_SIZE]
			output_logits = logits.eval(feed_dict=feed_dict)
			arg = tf.argmax(output_logits,1)
			for j in range(BATCH_SIZE):
				print('{0}: {1} - {2}', i*BATCH_SIZE+j, arg[j], filenames[j])

		print('Test of model')
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))

		"""
		print('Test model')
		test_accuracy = loss.eval(feed_dict={x:test['images'][0:BATCH_SIZE]})
		print('Test_accuracy={0:0.4f}'.format(test_accuracy))				
		"""

		
		"""
		# Saver
		saver = tf.train.Saver()		
		saver.save(sess, './save_model/my_test_model')  
		"""


