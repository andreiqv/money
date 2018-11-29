#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cut boxes by coordinates + shift.

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
	""" Returns: dict, where key is a line number in the file, 
			and value is a list of the following form: [class_id, x, y, w, h]
	"""

	boxes = dict()

	with open(txtfile_path, 'rt') as f:		
		for counter, line in enumerate(f):
			boxes[counter] = line.split()

			#class_id, x, y, w, h = line.split()			
			#boxes[counter] = {'class_id': class_id, 'x': x, 'y': y, 'w': w, 'h': h, '5':tuple5}

	return boxes


def cut_boxes(in_dir, out_dir):

	files = os.listdir(in_dir)

	count_intersection = 0
	count_shift_files = 0
	
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
				class_id, xr, yr, wr, hr = box  # "r" means relative units

				def to_absolute_value(xr, yr, wr, hr):				
					frame_width = 7 # frame width (to increase the window size)
					x = float(xr) * sx
					y = float(yr) * sy
					w = float(wr) * sx + frame_width
					h = float(hr) * sy + frame_width
					return x, y, w, h

				x, y, w, h = to_absolute_value(xr, yr, wr, hr)

				area = [x - w/2, y - h/2, x + w/2, y + h/2]
				if area[0] < 0: area[0] = 0
				if area[1] < 0: area[1] = 0
				if area[2] > sx: area[2] = sx
				if area[3] > sy: area[3] = sy

				print('{}: class_id={} x={:.2f} y={:.2f} w={:.2f} h={:.2f}'.\
					format(counter, class_id, x, y, w, h))

				#crop_and_save_image(out_filename, box)
				newbasename = '{0:07}'.format(index)
				box_filepath = out_dir + '/' + newbasename + '_' + str(counter) \
								+ '.' + class_id_maps_to_str[class_id] + '.jpg'

				
				img_box = img.crop(area)
				img_box.save(box_filepath)
				

				if class_id == '0': # money
					xshift = -(w+1) if x > sx/2 else w+1
					xnew = x + xshift
					ynew = y
					area = (xnew - w/2, ynew - h/2, xnew + w/2, ynew + h/2)
					print('New shifted position = ({:.2f}, {:.2f})'.format(xnew, ynew))
					box_filepath = out_dir + '/' + newbasename + '_shift_' + str(counter) \
								+ '.' + class_id_maps_to_str['1'] + '.jpg'
					
					intersection = False
					for i1 in boxes:
						class1, xr1, yr1, wr1, hr1 = boxes[i1]
						x1, y1, w1, h1 = to_absolute_value(xr1, yr1, wr1, hr1)
						if abs(xnew - x1) < w and abs(ynew - y1) < h:
							intersection = True							
							break

					if not intersection: 
						img_box = img.crop(area)						
						img_box.save(box_filepath)
						print('Saved frame ({:.2f}, {:.2f}) in {}'.format(xnew, ynew, box_filepath))
						count_shift_files += 1
					else:
						print('Intersection with the frame in ({:.2f}, {:.2f})!'.\
							format(x1, y1))
						count_intersection += 1					

			img.close()

	print('count_intersection:', count_intersection)
	print('count_shift_files:', count_shift_files)

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
	in_dir = '/home/andrei/Data/Datasets/Money/train/'
	out_dir = '/home/andrei/Data/Datasets/Money/cut2'	
	in_dir = in_dir.rstrip('/')
	out_dir = out_dir.rstrip('/')	
	os.system('mkdir -p {0}'.format(out_dir))

	cut_boxes(in_dir, out_dir)
