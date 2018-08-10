import os

def valid_weights_in_dir(in_dir):

	tmpfile = 'tmp.txt'
	datafile = 'data/money_nonb.data'

	files = os.listdir(in_dir)
	res_dict = dict()

	for index, filename in enumerate(files):

		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext == '.weights': continue
		network_name = '_'.join(base.split('_')[:-1])
		epoch = base.split('_')[-1]
		weights_filepath = in_dir + '/' + filename

		cmd = './darknet classifier valid {0} cfg/{1}.cfg {2} > {3}'\
			.format(datafile, network_name, weights_filepath, tmpfile)

		print(cmd)
		os.system(cmd)

		f = open(tmpfile)
		res = f.readlines()[-1].strip()
		res_dict[epoch] = res
		f.close()


	print('\n\nRESULTS:')
	for epoch in res_dict:
		print('{0}: - {1}'.format(epoch, res_dict[epoch]))



if __name__ == '__main__':

	in_dir = 'backup'
	valid_weights_in_dir(in_dir)