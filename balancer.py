#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
INPUT FORMAT:  class x y w h
<class-id> <x> <y> <width> <height>
0 0.2123 0.2371 0.7735 0.9142
0 0.2638 0.3006 0.8056 0.9155

OUTPUT:
originalfilename_counter.class.jpg
1) class = money - if class_id = 0
2) class = err - if class_id = 1


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
#import numpy as np
import random

def parse_txtfile(txtfile_path):

	boxes = dict()

	with open(txtfile_path, 'rt') as f:		
		for counter, line in enumerate(f):
			#class_id, x, y, w, h = line.split()			
			#boxes[counter] = {'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h, '5':tuple5}
			boxes[counter] = line.split()

	return boxes


def get_files_list(in_dir):

	files = os.listdir(in_dir)

	dict_counter = {'money':0, 'err':0}
	dict_lists = {'money': [], 'err':[]}

	mapsl_class_to_str = {'0': 'money', '1': 'err'}
	maps_str_to_id = {'money': 0, 'err': 1}
	
	for filename in files:
		#base = os.path.splitext(in_file_name)[0]
		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]

		if not ext == '.jpg': 
			continue

		class_str = base.split('_')[-1]
		dict_counter[class_str] += 1
		dict_lists[class_str].append(filename)

		class_id = maps_str_to_id[class_str]
		print('base={0}, class_str={1}, class_id={2}'.format(base, class_str, class_id))

		
		#os.system('mv {0} {1}'.format(jpg_file_old_path, jpg_file_new_path))
		#print('{0} -> {1}'.format(jpg_file_old_path, jpg_file_new_path))

	#print(dict_counter)

	for class_str in dict_lists:
		random.shuffle(dict_lists[class_str])
		print(dict_lists[class_str])

	return dict_lists


def copy_files_balanced(in_dir, out_dir, dict_lists, train_percent=0.8):
	# copy images to directories train and valid:
	
	minsize = min({ len(dict_lists[class_str]) for class_str in dict_lists })
	print('minsize =', minsize)

	for i in range(minsize):
		print('\ni =', i)
		for class_str in dict_lists:
			filename = dict_lists[class_str][i]
			in_file = in_dir+'/'+filename
			if i < minsize * train_percent:
				out_file = out_dir + '/train/' + filename 
			else:
				out_file = out_dir + '/valid/' + filename 
			cmd = 'cp {0} {1}'.format(in_file, out_file)
			print(cmd)			
			os.system(cmd)



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

	#in_dir = '/mnt/work/ineru/data/train'
	in_dir = '/w/WORK/ineru/04_money/rep_money/examples'
	out_dir = '/w/WORK/ineru/04_money/rep_money/examples_balanced'
	
	#in_dir = '/home/chichivica/Data/Datasets/Money/cut'
	#out_dir = '/home/chichivica/Data/Datasets/Money/balanced'	
	
	in_dir = in_dir.rstrip('/')
	out_dir = out_dir.rstrip('/')
	os.system('mkdir -p {0}'.format(out_dir))
	os.system('mkdir -p {0}'.format(out_dir + '/train'))
	os.system('mkdir -p {0}'.format(out_dir + '/valid'))

	dict_lists = get_files_list(in_dir)
	copy_files_balanced(in_dir, out_dir, dict_lists, train_percent=0.6)

	# find /mnt/work/ineru/data/train -type f