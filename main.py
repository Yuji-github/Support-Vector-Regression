import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer # for encoding categorical to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting
from sklearn.preprocessing import PolynomialFeatures # for polynomial linear regression
from sklearn.svm import SVR

def SVR():
    # import data
    data = pd.read_csv('Position_Salaries.csv')
    # print(data)

    # independent variable
    x = data.iloc[:, 1:-1].values

    # dependent variable
    y = data.iloc[:, -1].values

    # print(x, y)
    plt.scatter(x, y, color='orange', marker='X')
    plt.xlabel('Position')
    plt.ylabel('Salary')
    plt.show()

    # missing data
    # impute = SimpleImputer()
    # impute.fit(x)
    # x = impute.transform(x)

    # splitting data into 4 parts ***Skip this processing as the data is too small 10***
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

    # feature scaling
    # 1: SVR is not explicit equation
    # 2: Both feature value is different (10 vs 1000000), so MUST apply feature scaling dependent and independent variables
    # *** even a equation is  implicit equation like the multiple linear regression, MUST follow the second condition

    y = y.reshape(len(y), 1) # need to reshape as StandardScalar expects 2D array
    # sc_x is only for x as it return Standard Deviation of x
    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)

    # sc_y is only for y as it return Standard Deviation of y
    sc_y = StandardScaler()
    y = sc_y.fit_transform(y)
    # print(x, y)

    # training data https://data-flair.training/blogs/svm-kernel-functions/
    svr_regressor = SVR(kernel='rbf') # default is kernel='rbf' this works on Google Colab: Not local PC because of the kernel
    svr_regressor.fit(x, y)
    
    # predicting 6.5 years
    print('\n6.5 years of Salary is %d dollars' %sc_y.inverse_transform(svr_regressor.predict(sc_x.transform([[6.5]]))))

    x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1) # for smooth lines
    x_grid = x_grid.reshape(len(x_grid), 1) # for smooth lines
    plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red', marker='X')
    plt.plot(x_grid, sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(x_grid))), color='blue')
    plt.title('Support Vector Regression')
    plt.xlabel('Position')
    plt.ylabel('Salary')
    plt.show()

if __name__ == '__main__':
    SVR()
