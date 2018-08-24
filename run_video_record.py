import os

def run_record(in_dir):
	
	#datafile = 'data/money_nonb.data'

	files = os.listdir(in_dir)
	files.sort()

	for index, filename in enumerate(files):

		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.mp4', ".avi"} : continue

		in_file = in_dir + '/' + filename
		out_file = base + '.avi'

		#cmd = './darknet classifier valid {0} cfg/{1}.cfg {2} > {3}'\
		#	.format(datafile, network_name, weights_filepath, tmpfile)

		cmd = "./darknet detector demo data/money.data cfg/yolov3-money.cfg"\
			 " ../yolov3-money_40000.weights {0} -thresh 0.1 -i 1 -prefix {1}"\
			 .format(in_file, out_file)

		print(cmd)
		os.system(cmd)



if __name__ == '__main__':

	in_dir = '/home/chichivica/Data/Datasets/Money/raw_video/1/'
	run_record(in_dir)