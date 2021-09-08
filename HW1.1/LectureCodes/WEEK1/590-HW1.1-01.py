import json
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize,fmin_tnc
import pandas as pd
from scipy.special import expit

class Data:
    def __init__(self,filename):
        self.filename = filename

    def read_data(self):
        with open(self.filename) as f:
            self.data = json.load(f)
            return self.data

    def partition_data(self):
        self.df = pd.DataFrame(data)
        return self.df

    def visualize_data(self):
        x = input('x axis colname')
        y = input('y axis colname')
        plt.plot(self.df[x],self.df[y])
        plt.show()

weight= Data('weight.json')
data = weight.read_data()
df = weight.partition_data()
print("ready to visualize data, for this case, please enter 'x' for the colname of x, and 'y' for the colname of y")
weight.visualize_data()

## ======== Linear Regression ===============
#y = np.array(data['is_adult'])

#X = np.array(data['x'])

#df = pd.DataFrame(data, columns=['x','y','is_adult'])

## dataset for age under 18
df_18 = df[df['x']<18]

## normalization, train test split
standard_scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(np.array(df_18['x']), np.array(df_18['y']), test_size = 0.2,random_state=42)

X_train_norm = standard_scaler.fit_transform(X_train.reshape(-1,1))
X_test_norm = standard_scaler.transform(X_test.reshape(-1,1))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


## model and cost function
def f(x, p):
    y = p[0]+x*p[1]
    return y

def obj(p):
    yp = f(X_train_norm,p)
    loss = np.mean((y_train - yp)**2)
    return loss

## optimization
NFIT = 2
po = np.random.uniform(0.5, 1., size=NFIT)
# res.x contains your coefficients
#res = minimize(lambda coeffs: obj(X_train_norm, y_train, *coeffs), x0=po,method='Nelder-Mead')

res = minimize(obj, x0=po,method='SLSQP',tol=1e-15)
print(res.x)

## plotting
fig, ax = plt.subplots()
#ax.plot(df['x'],df['y'],'o',X_train,f(X_train_norm,res.x),'-')
ax.plot(df['x'],df['y'],'.', markersize=12,color='blue',label="raw data") # ,color='black', markersize=8)
ax.plot(X_train,f(X_train_norm,res.x),'r-',linewidth=3,label="model")
ax.plot(X_test,f(X_test_norm,res.x),'*',color='green',label="prediction")
ax.plot(X_test,y_test,'.', markersize=12,color='yellow',label="testset")
ax.legend()
plt.show()




## ======== Logistic Regression ===============

## normalization, train test split
standard_scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(np.array(df['x']), np.array(df['y']), test_size = 0.2,random_state=42)

X_train_norm = standard_scaler.fit_transform(X_train.reshape(-1,1))
X_test_norm = standard_scaler.transform(X_test.reshape(-1,1))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

def f_sigmoid(x,p):
    A = p[0]
    w = p[1]
    x0 = p[2]
    S=p[3]
    y = (A/(1 + expit(((-(x-x0))/w))))+S
    return y
def obj(p):
    yp = f_sigmoid(X_train_norm,p)
    loss = np.mean((y_train - yp)**2)
    return loss

# def obj_logistic(p):
#     #epsilon = 1e-5
#     #y = y_train
#     #x = X_train_norm
#     #loss = np.mean(-y*(np.log(f_sigmoid(x,p)+epsilon)) - (1-y)*np.log(1-f_sigmoid(x,p)+epsilon))
#     yp = f_sigmoid(X_train_norm,p)
#     print((yp-y_train)**2)
#     loss = np.mean((y_train - yp)**2)
#     print(loss)
#     return loss




NFIT = 4
po=np.random.uniform(1,10,size=NFIT)
res = minimize(obj, po, method = 'Nelder-Mead',tol=1e-8)
popt=res.x
print("OPTIMAL PARAM:",popt)

fig, ax = plt.subplots()
ax.plot(df['x'],df['y'],'o',X_train,f_sigmoid(X_train_norm,res.x))
plt.show()


## ======== Logistic Regression II ===============

## normalization, train test split
standard_scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(np.array(df['y']), np.array(df['is_adult']), test_size = 0.2,random_state=42)

X_train_norm = standard_scaler.fit_transform(X_train.reshape(-1,1))
X_test_norm = standard_scaler.transform(X_test.reshape(-1,1))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

def f_sigmoid(x,p):
    A = p[0]
    w = p[1]
    x0 = p[2]
    S=p[3]
    y = (A/(1 + expit(((-(x-x0))/w))))+S
    return y
def obj(p):
    yp = f_sigmoid(X_train_norm,p)
    loss = np.mean((y_train - yp)**2)
    return loss

# def obj_logistic(p):
#     #epsilon = 1e-5
#     #y = y_train
#     #x = X_train_norm
#     #loss = np.mean(-y*(np.log(f_sigmoid(x,p)+epsilon)) - (1-y)*np.log(1-f_sigmoid(x,p)+epsilon))
#     yp = f_sigmoid(X_train_norm,p)
#     print((yp-y_train)**2)
#     loss = np.mean((y_train - yp)**2)
#     print(loss)
#     return loss




NFIT = 4
po=np.random.uniform(1,10,size=NFIT)
res = minimize(obj, po, method = 'Nelder-Mead',tol=1e-8)
popt=res.x
print("OPTIMAL PARAM:",popt)

fig, ax = plt.subplots()
ax.plot(df['y'],df['is_adult'],'o',X_train,f_sigmoid(X_train_norm,res.x))
plt.show()



