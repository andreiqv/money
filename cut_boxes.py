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

from PIL import Image



def parse_txtfile(txtfile_path):

	boxes = dict()

	with open(txtfile_path, 'rt') as f:		
		for counter, line in enumerate(f):
			#class_id, x, y, w, h = line.split()			
			#boxes[counter] = {'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h, '5':tuple5}
			boxes[counter] = line.split()

	return boxes


def cut_boxes(in_dir, out_dir):

	files = os.listdir(in_dir)
	
	for index, filename in enumerate(files):
		#base = os.path.splitext(in_file_name)[0]
		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]

		if not ext == '.jpg': 
			continue

		txtfile = base + '.txt'
		jpgfile = filename

		jpgfile_path = in_dir + '/' + jpgfile
		txtfile_path = in_dir + '/' + txtfile

		if  not os.path.exists(txtfile_path):
			print('txt file {0} does not exist.'.format(txtfile_path))
			raise Exception('txt file does not exist.')

		print('\n', txtfile_path)	
		boxes = parse_txtfile(txtfile_path)

		class_id_maps_to_str = {'0': 'money', '1': 'err'}

		if len(boxes) > 0:

			img = Image.open(jpgfile_path)
			sx, sy = img.size

			for counter in boxes:
				box = boxes[counter]
				class_id, x, y, w, h = box
				x = float(x) * sx
				y = float(y) * sy
				w = float(w) * sx
				h = float(h) * sy
				area = (x - w/2, y - h/2, x + w/2, y + h/2)

				print('{0}: class_id={1} x={2} y={3} w={4} h={5}'.\
					format(counter, class_id, x, y, w, h))

				#crop_and_save_image(out_filename, box)
				newbasename = '{0:07}'.format(index)
				box_filepath = out_dir + '/' + newbasename + '_' + str(counter) \
								+ '.' + class_id_maps_to_str[class_id] + '.jpg'
				#box_filepath = out_dir + '/' + base + '_' + str(counter) \
				#				+ '.' + class_id_maps_to_str[class_id] + '.jpg'
				
				img_box = img.crop(area)
				img_box.save(box_filepath)
			
			img.close()

		
		#convert_file(in_file_path, out_file_path)

		#os.system('mv {0} {1}'.format(jpg_file_old_path, jpg_file_new_path))
		#print('{0} -> {1}'.format(jpg_file_old_path, jpg_file_new_path))

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

	#in_dir = 'images'
	#out_dir = 'out'
	in_dir = '/home/chichivica/Data/Datasets/Money/train'
	out_dir = '/home/chichivica/Data/Datasets/Money/cut'	
	in_dir = in_dir.rstrip('/')
	out_dir = out_dir.rstrip('/')	
	os.system('mkdir -p {0}'.format(out_dir))

	cut_boxes(in_dir, out_dir)
