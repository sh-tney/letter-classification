import requests
import os.path
import sys
import pickle, gzip
import tensorflow as tf
import numpy as np

def download_file(filename, url):
   """
   Download an URL to a file
   """
   with open(filename, 'wb') as fout:
      response = requests.get(url, stream=True)
      response.raise_for_status()
      # Write response data to file
      for block in response.iter_content(4096):
         fout.write(block)


def download_if_not_exists(filename, url):
   """
   Download a URL to a file if the file
   does not exist already.
   Returns
   -------
   True if the file was downloaded,
   False if it already existed
   """
   if not os.path.exists(filename):
      sys.stdout.write("Downloading data file...")
      sys.stdout.flush()
      download_file(filename, url)
      sys.stdout.write("done\n")
      sys.stdout.flush()
      return True
   return False


def load_data(task='simple',dtype='clean'):

   if task != 'simple' and task != 'fine-grained' and task != 'multi-label':
      raise ValueError("Invalid value for the 'task' argument (valid options are: 'simple','fine-grained', or 'multi-label')")

   if dtype != 'clean' and dtype != 'noisy':
      raise ValueError("Invalid value for the 'dtype' argument (valid options are: 'clean', or 'noisy')")

   if task == 'simple' or task == 'fine-grained':
      if dtype == 'clean':
         filename = 'data_alpha_rot_20000.gz.pickle'
         url = 'https://dl.dropbox.com/s/qqpmuvto3yz0jfz/data_alpha_rot_20000.gz.pickle?dl=1'
      else:
         filename = 'data_alpha_fc_bc_rot_60000.gz.pickle'
         url = 'https://dl.dropbox.com/s/2k86dzkwxlvyjjf/data_alpha_fc_bc_rot_60000.gz.pickle?dl=1'
   else:
      if dtype == 'clean':
         filename = 'data_alpha_rot_ml_mc_20000.gz.pickle'
         url = 'https://dl.dropbox.com/s/rvzcdwoqtmqbqcd/data_alpha_rot_ml_mc_20000.gz.pickle?dl=1'
      else:
         filename = 'data_alpha_fc_bc_rot_ml_mc_60000.gz.pickle'
         url = 'https://dl.dropbox.com/s/2gqxo25vlnm8dhy/data_alpha_fc_bc_rot_ml_mc_60000.gz.pickle?dl=1'

   basedir = 'data'

   if not os.path.isdir(basedir):
      os.mkdir(basedir)

   filename = os.path.join('data',filename)

   download_if_not_exists(filename,url)


   with gzip.open(filename) as f:
      (train_images, train_labels), (test_images, test_labels), class_names, (train_font_labels, test_font_labels), font_names = pickle.load(f)

   for i in range(len(font_names)):
        font_names[i] = font_names[i][:-4]

   if task=='multi-label':
        train_font_labels = tf.keras.utils.to_categorical(train_font_labels)
        test_font_labels = tf.keras.utils.to_categorical(test_font_labels)
        train_labels = np.concatenate((train_labels,train_font_labels),axis=1)
        test_labels = np.concatenate((test_labels,test_font_labels),axis=1)
        class_names += font_names

   if task=='fine-grained':
        class_names = font_names
        train_labels = train_font_labels
        test_labels = test_font_labels


   return (train_images, train_labels), (test_images, test_labels), class_names


if __name__ == "__main__":
	task = 'simple'
	dtype = 'clean'

	# Test code for loading the data and showing the first 16 train images
	(train_images, train_labels), (test_images, test_labels), class_names = load_data(task='simple', dtype='clean')

	import os
	if os.path.exists('show_methods.py'):
		import matplotlib.pyplot as plt
		import show_methods

		show_methods.show_data_images(images=train_images[:16],labels=train_labels[:16],class_names=class_names)
		print(task + dtype + '.png')
		plt.savefig(task + dtype + '.png')