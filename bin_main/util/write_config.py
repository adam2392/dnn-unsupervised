import numpy as np 
import sys
import json
import pandas as pd 
import datetime

NOW = datetime.datetime.now()

class ConfigWriter(object):

	def __init__(self, root_dir, expname, filename):
		self.root_dir = root_dir

		# create temporary directory
	    tempdir = os.path.join(root_dir, expname)
	    if not os.path.exists(tempdir):
	        os.makedirs(tempdir)

	    # write these to config file
	    config_file = os.path.join(tempdir, filename)

	    output = open(config_file, "w")
	    self.config_file = output
	
	def _write_config_row(self, row):
		self.config_file.write(row + '\n')


    def write_exp_config(self, patient, traindatadir, testdatadir, num_epochs):
    	config_text = []
    	# create the row object
        row = patient + ',' + \
              traindatadir + ',' + \
              testdatadir + ',' + \
              str(num_epochs) + ',' + \
              str(NOW)
        config_text.append(row)
            
        for row in config_text:
        	self._write_config_row(row)

        self.config_file.close()

if __name__ == '__main__':
	patient = 'id001_bt'
	traindatadir = '.'
	testdatadir = '.'
	num_epochs = 100
