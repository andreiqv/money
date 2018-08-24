import os
import sys
import argparse

def run_record(in_dir, i):
	
	#datafile = 'data/money_nonb.data'

	files = os.listdir(in_dir)

	for index, filename in enumerate(files):

		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.mp4', ".avi"} : continue

		in_file = in_dir + '/' + filename
		out_file = base + '.avi'

		#cmd = './darknet classifier valid {0} cfg/{1}.cfg {2} > {3}'\
		#	.format(datafile, network_name, weights_filepath, tmpfile)

		cmd = "../darknet-fork/darknet detector demo ../money.data ../yolov3-money.cfg"\
			 " ../yolov3-money_40000.weights {0} -thresh 0.1 -i {1} -prefix {2}"\
			 .format(in_file, i, out_file)

		print(cmd)
		os.system(cmd)


#---
def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--i', default=1, type=int,\
		help='gpu')
	#parser.add_argument('-phi', '--phi', dest='phi', action='store_true')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	print('set arguments.i =', arguments.i)

	i = arguments.i
	in_dir = '/home/chichivica/Data/Datasets/Money/raw_video/{0}'.format(i)
	run_record(in_dir, i)