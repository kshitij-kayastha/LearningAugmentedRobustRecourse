import torch
import numpy as np 
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
from abc import ABC, abstractmethod



class Model(ABC):
	def __init__(self):
		pass

	def metrics(self, X, y):
		acc = np.mean(self.predict(X)==y)

		pred = self.predict_proba(X)[:,1]
		fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
		auc = metrics.auc(fpr, tpr)

		return acc, auc
 
	@abstractmethod
	def train(self, X, y):
		pass



class LR(Model):
	def __init__(self):
		super(LR, self).__init__()

	def train(self,X,y):
		model = LogisticRegression().fit(X, y)
		self.model = model

		self.weights = torch.from_numpy(model.coef_[0]).float()
		self.bias = torch.tensor(model.intercept_[0]).float()

	def torch_model(self, x):
		return torch.nn.Sigmoid()(torch.matmul(self.weights, x) + self.bias)[0]

	def predict(self, x):
		return self.model.predict(x)

	def predict_proba(self, x):
		return self.model.predict_proba(x)


class NN(Model):
	def __init__(self, num_feat):
		torch.manual_seed(0)
		super(NN, self).__init__()
		self.model = nn.Sequential(
		  nn.Linear(num_feat, 50),
		  nn.ReLU(),
		  nn.Linear(50, 100),
		  nn.ReLU(),
		  nn.Linear(100, 200),
		  nn.ReLU(),
		  nn.Linear(200, 1),
		  nn.Sigmoid()
		  )
	
	def torch_model(self,x):
		return self.model(x)[0]

	def train(self, X_train, y_train, verbose=0):
		torch.manual_seed(0)
		X_train = torch.from_numpy(X_train).float()
		y_train = torch.from_numpy(y_train).float()

		loss_fn = nn.BCELoss()
		optimizer = torch.optim.Adam(self.model.parameters())

		# Train model
		epochs = 100
		for epoch in range(epochs):
			self.model.train()
			optimizer.zero_grad()

			# Forward pass
			y_pred = self.model(X_train)
			# Compute Loss
			loss = loss_fn(y_pred[:,0], y_train)
			if verbose: print(f'Epoch {epoch}: train loss: {loss.item()}')

			# Backward pass
			loss.backward()
			optimizer.step()

	def predict_proba(self, X):
		X = torch.from_numpy(np.array(X)).float()
		class1_probs = self.model(X).detach().numpy()
		class0_probs = 1-class1_probs
		return np.hstack((class0_probs,class1_probs))

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)
