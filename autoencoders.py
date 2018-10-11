import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing the dataset
movies = pd.read_csv('ml-lm/movies.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')
users = pd.read_csv('ml-lm/users.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')
rating = pd.read_csv('ml-lm/ratings.dat',sep='::',header = None,engine = 'python',encoding = 'latin-1')

#preparing the training set and test set
training_set = pd.read_csv('ml-100k/ul.base',delimiter = '\t')
training_set = np.array(training_set,dtype = 'int')
test_set = pd.read_csv('ml-100k/ul.test',delimiter = '\t')
test_set = np.array(test_set,dtype='int')

#put all users and movies into an array with all user id and movie ids with the value of rating,
# If the user haven't rate the movie, the value is 0

#Getting the number of total users and total movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

#Converting the data into an array with users in rows and movies in columns
# the array is a list of list, which is the data structure needed to transfer into torch sensor
def convert(data):
	new_data = []
	for id_users in range(1,nb_users+1):
		id_movies = data[:,1][data[:,0] == id_users]
		id_ratings = data[:,2][data[:,0] == id_users]
		ratings = np.zeros(nb_movies)
		ratings[id_movies-1] = id_ratings
		new_data.append(lists(ratings))
	return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#convert the data into torch tensors
training_set = torch.FloatTensor(trainig_set)
test_set = torch.FloatTensor(test_set)














