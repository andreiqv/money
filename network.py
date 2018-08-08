#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
"""
# export CUDA_VISIBLE_DEVICES=1

from __future__ import absolute_import,  division, print_function
import tensorflow as tf
#import tensorflow_hub as hub
import sys
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from layers import *


def conv_network_1(x_image, output_size=2):
	# input 64 x 64 x 3
	# conv layers 
	p1 = convPoolLayer(x_image, kernel=(5,5), pool_size=2, num_in=3, num_out=16, 
		func=tf.nn.relu, name='1') # 32 x 32 x 16
	p2 = convPoolLayer(p1, kernel=(5,5), pool_size=2, num_in=16, num_out=16, 
		func=tf.nn.relu, name='2')  # 16 x 16 x 16
	p3 = convPoolLayer(p2, kernel=(4,4), pool_size=2, num_in=16, num_out=32, 
		func=tf.nn.relu, name='3')   # 8 x 8 x 32
	p4 = convPoolLayer(p3, kernel=(3,3), pool_size=2, num_in=32, num_out=32, 
		func=tf.nn.relu, name='4')   # 4 x 4 x 32

	# fully-connected layers
	fullconn_input_size = 4*4*32
	p_flat = tf.reshape(p4, [-1, fullconn_input_size])

	f1 = fullyConnectedLayer(p_flat, input_size=fullconn_input_size, num_neurons=1024, 
		func=tf.nn.relu, name='F1')

	drop1 = tf.layers.dropout(inputs=f1, rate=0.4)	
	
	f2 = fullyConnectedLayer(drop1, input_size=1024, num_neurons=256, 
		func=tf.nn.relu, name='F2')
	
	drop2 = tf.layers.dropout(inputs=f2, rate=0.4)

	f3 = fullyConnectedLayer(drop2, input_size=256, num_neurons=output_size, 
		func=None, name='F3')	 # it doesn't work with sigmoid or relu

	return f3

#--------




