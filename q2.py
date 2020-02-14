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
print("\n BIAS^2 VS VARIANCE\n")
print("#"*80)

class linear_regression:
    def __init__(self):
        self.test_data_x = []
        self.test_data_y = []
        self.split_train_data_x = []
        self.split_train_data_y = []
        self.split_test_data_x = []
        self.split_test_data_y = []
        self.bias = []
        self.variance = []
        self.b_final = []
        self.v_final = []
# ---------------------------Data Refactoring-----------------------------------
    def b_v_averaging(self,b,v):
        for i in range(10):
            b=np.asarray(b)
            bias_total=b[i*20:(i+1)*20].mean()
            v=np.asarray(v)
            variance_total=v[i*20:(i+1)*20].mean()
            self.b_final.append(bias_total)
            self.v_final.append(variance_total)
            print(bias_total," | ",variance_total)

    def data_refactoring(self):
        Y_test = open('./Q2_data/Fx_test.pkl', 'rb')
        X_test = open('./Q2_data/X_test.pkl','rb')
        Y_train = open('./Q2_data/Y_train.pkl','rb')
        X_train = open('./Q2_data/X_train.pkl','rb') 
        self.split_test_data_x = pickle.load(X_test)
        self.split_test_data_y = pickle.load(Y_test)
        self.split_train_data_x = pickle.load(X_train)
        self.split_train_data_y = pickle.load(Y_train)
        # print(self.split_test_data_x)
# ---------------------------Model Training-------------------------------------
    def plot_check(self, x, y):
            plt.plot(range(10),x)
            plt.title('number of parameters vs Bias')
            plt.xlabel('Number of parameters')
            plt.ylabel('Bias^2')
            plt.show()
            plt.plot(range(10),y)
            plt.title('number of parameters vs Variance') 
            plt.xlabel('Number of parameters')
            plt.ylabel('Variance')
            plt.show()

    def model_training(self):
        for i in range(10):
            for j in range(20):
                self.test_data_x = self.split_test_data_x
                self.test_data_y = self.split_test_data_y
                model = lr()
                poly=PolynomialFeatures(degree=i)
                x=self.split_train_data_x[j][...,np.newaxis]
                y=self.split_train_data_y[j][..., np.newaxis]
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
        for i in range(80):
            bias = (y_predicted[i] - y_test[i])**2
            bias_total += bias
        variance_total = statistics.variance(y_predicted)
        bias_total /= 80
        # print(j,"degree :-",bias_total," | ",variance_total)
        self.bias.append(bias_total)
        self.variance.append(variance_total)
# ---------------------------Driver Function------------------------------------
def main():
    ob=linear_regression()
    ob.data_refactoring()
    ob.model_training()
    ob.b_v_averaging(ob.bias,ob.variance)
    ob.plot_check(ob.b_final,ob.v_final)

if __name__ == '__main__':
    main()

