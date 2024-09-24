# LogisitcRegression.py
import pandas as pd # type: ignore
import numpy as np # type: ignore 
import matplotlib.pyplot as plt # type: ignore
import warnings
import seaborn as sns # type: ignore
from tqdm import tqdm # type: ignore

warnings.filterwarnings('ignore')


class LogitRegression():
    def __init__(self,learning_rate,iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weight = None
        self.bias = None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def cost(self,X,y):
        z = np.dot(X,self.weight) + self.bias
        h = self.sigmoid(z)
        n=y.shape[0]
        cost = -(1/n)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
        # print(cost,np.log(1-h)[0],h[0],z[0],X[0],self.weight[0],self.bias)
        return cost
    
    def gradient_descent(self,X,y):
        N = y.shape[0]
        z = np.dot(X,self.weight) + self.bias
        pred = self.sigmoid(z)
        delta = pred - y
        dW = (1/N)*np.dot(X.T,delta)
        dB = (1 / N) * np.sum(delta)
        self.weight -=self.learning_rate*dW
        self.bias -=  self.learning_rate*dB


    def fit(self,X,y):
        loss = []
        N,numFeatures = X.shape
        self.weight = np.random.uniform(0, 1, numFeatures)
        self.bias = 0.0
        for i in tqdm(range(self.iterations)):
            self.gradient_descent(X,y)
            loss.append(self.cost(X,y))
        return loss



    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        y_predicted = self.sigmoid(z)
        y_predicted_class = [1 if i >= 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

