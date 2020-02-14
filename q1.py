import pickle
import numpy as np
from sklearn.model_selection import train_test_split as tts, ShuffleSplit 
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import math
import statistics
print("\n\n\n")
print("#"*80)
print("\n BIAS VS VARIANCE\n")
print("#"*80)

class linear_regression:
    def __init__(self):
        self.train_data = None
        self.train_data_x = None
        self.train_data_y = None
        self.test_data = None
        self.test_data_x = None
        self.test_data_y = None
        self.split_train_data = []
        self.bias = []
        self.variance = []
# ---------------------------Data Refactoring-----------------------------------
    def chunk_up_split(self,seq,num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out
 
    def data_refactoring(self):
        pkl_file = open('./Q1_data/data.pkl', 'rb')
        net_data = pickle.load(pkl_file)
        self.train_data, self.test_data= tts(net_data, test_size=0.1, train_size=0.9, shuffle=True)
        self.train_data_x=self.train_data[:,0]
        self.train_data_y=self.train_data[:,1]
        self.test_data_x=self.test_data[:,0]
        self.test_data_y=self.test_data[:,1]
        self.split_train_data_x = self.chunk_up_split(self.train_data_x, 10)
        self.split_train_data_y = self.chunk_up_split(self.train_data_y, 10)
# ---------------------------Model Training-------------------------------------
    def plot_check(self, x, y):
            plt.plot(range(10),x)
            plt.title('number of parameters vs Bias')
            plt.xlabel('Number of parameters')
            plt.ylabel('Bias')
            plt.show()
            plt.plot(range(10),y)
            plt.title('number of parameters vs Variance') 
            plt.xlabel('Number of parameters')
            plt.ylabel('Variance')
            plt.show()

    def model_training(self):
        for i in range(10):
            model = lr()
            poly=PolynomialFeatures(degree=i)
            x=self.split_train_data_x[i][...,np.newaxis]
            y=self.split_train_data_y[i][..., np.newaxis]
            x_=poly.fit_transform(x)
            x_test=poly.fit_transform(self.test_data_x[...,np.newaxis])
            model.fit(x_, y)
            predicted_y=model.predict(x_test)
            plt.plot(self.test_data_x[...,np.newaxis],self.test_data_y[...,np.newaxis],'o')
            plt.title('X vs Y')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(self.test_data_x,predicted_y.flatten(), 'o', color='black')
            plt.show()
            self.bias_variance_calculation(self.test_data_x, self.test_data_y,predicted_y.flatten(),i)
# ---------------------------Bias Variance-----------------------------------
    def bias_variance_calculation(self, x_test, y_test, y_predicted, j):
        bias_total = 0
        variance_total = 0
        E_y_predicted = y_predicted.mean()
        for i in range(500):
            bias = (E_y_predicted - y_test[i])**2
            bias_total += bias
        variance_total = statistics.variance(y_predicted)
        bias_total /= 500
        print(j,"degree :-",bias_total," | ",variance_total)
        self.bias.append(bias_total)
        self.variance.append(variance_total)
# ---------------------------Driver Function------------------------------------
def main():
    ob=linear_regression()
    ob.data_refactoring()
    ob.model_training()
    ob.plot_check(ob.bias,ob.variance)

if __name__ == '__main__':
    main()

