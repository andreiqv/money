#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""


"""
from __future__ import division  
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import argparse
import math
import logging
from collections import namedtuple
import operator # for min
import numpy as np
import random
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip

import balancer

def load_data_with_lists(in_dir, dict_lists, shape=(64,64,3)):
	"""
	in_dir - input dir
	dict_lists - dict of lists of files
	"""

	#map_id_str = {0: 'money', 1:'err'}
	num_classes = 2
	dict_lists[0] = dict_lists['money']
	dict_lists[1] = dict_lists['err']
	count = [0, 0]
	minsize = min(len(dict_lists[0]), len(dict_lists[1]))
	print('size_0 = {0}, size_1 = {1}'.format(len(dict_lists[0]), len(dict_lists[1])))

	data = dict()
	data['filenames'] = []
	data['images'] = []
	data['labels'] = []	
	img_size = shape[0], shape[1]

	while count[0] < minsize or count[1] < minsize:
		class_id = random.randint(0,1)
		if count[class_id] == minsize:
			continue			
		#print('{0}: count={1}'.format(class_id, count[class_id]))
		filename = dict_lists[class_id][count[class_id]]
		print('{0}: count={1}, {2}'.\
			format(class_id, count[class_id], filename))
		count[class_id] += 1

		filepath = in_dir + '/' + filename
		img = Image.open(filepath)
		img = img.resize(img_size, Image.ANTIALIAS)
		#img.save(in_dir + '/' + 'resized_' + filename)
		arr = np.array(img, dtype=np.float32) / 256
		if class_id==0:
			lable = np.array([1, 0], dtype=np.float32)
		elif class_id==1:
			lable = np.array([0, 1], dtype=np.float32)
		else:
			raise Exception('bad class_id')
		data['images'].append(arr)
		data['labels'].append(lable)
		data['filenames'].append(filename)

	print('Stop: count_0={0}, count_1={1}'.format(count[0], count[1]))

	return data
	




def split_data(data, ratio):

	len_data = len(data['labels'])
	assert len_data == len(data['labels'])

	len_train = len_data * ratio[0] // sum(ratio)
	len_valid = len_data * ratio[1] // sum(ratio)
	len_test  = len_data * ratio[2] // sum(ratio)
	print(len_train, len_valid, len_test)

	splited_data = {'train': dict(), 'valid': dict(), 'test': dict()}
	
	for key in data:
		splited_data['train'][key] = data[key][ : len_train]
		splited_data['valid'][key] = data[key][len_train : len_train + len_valid]
		splited_data['test'][key] = data[key][len_train + len_valid : ]

	for key in splited_data:
		splited_data[key]['size'] = len(splited_data[key]['labels'])

	return splited_data
#---------------

def createParser ():
	"""	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	#parser.add_argument('-th', '--threshold', default=0.05, type=float,\
	#	help='threshold value (default 0.05)')
	#parser.add_argument('-df', '--diff', dest='diff', action='store_true')

	return parser


if __name__ == '__main__':	

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])	
	#threshold = arguments.threshold	

	in_dir = '/mnt/work/ineru/data/train'
	#in_dir = '/w/WORK/ineru/04_money/rep_money/examples'
	#out_dir = '/w/WORK/ineru/04_money/rep_money/examples_balanced'
	#in_dir = '/home/chichivica/Data/Datasets/Money/cut'
	dump_file = 'dump.gz'	
	
	in_dir = in_dir.rstrip('/')
	dict_lists = balancer.get_files_list(in_dir)
	data = load_data_with_lists(in_dir, dict_lists, shape=(64,64,3))
	data = split_data(data, ratio=(7,1,2))
	dump = pickle.dumps(data)
	print('dump.pickle')
	with gzip.open(dump_file, 'wb') as f:
		f.write(dump)
		print('gzip dump was written')	