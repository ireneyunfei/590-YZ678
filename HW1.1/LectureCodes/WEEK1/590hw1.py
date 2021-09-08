import json
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize,fmin_tnc

with open('weight.json') as f:
  data = json.load(f)

y = np.array(data['is_adult'])

X = np.array(data['x'])

standard_scaler = preprocessing.StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

X_train_minmax = standard_scaler.fit_transform(X_train.reshape(-1,1))
X_test_minmax = standard_scaler.transform(X_test.reshape(-1,1))


def f(x, a, b):
    y = a*x+b
    return y

def obj(x, y, a, b):
    loss = np.sum((y - f(x,a,b))**2)
    return loss

#bounds = [(None, None), (None, 0)]

# res.x contains your coefficients
res = minimize(lambda coeffs: obj(X, y, *coeffs), x0=np.zeros(2), bounds=bounds)

import numpy as np
import math
import matplotlib.pyplot as plt
x1 = np.arange(-0.8, 2.3, 0.1)
y1 = []
for t in x1:
    y_1 = f_sigmoid(t,a,b,c,d)
    y1.append(y_1)
plt.plot(x1, y1, label="sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(0, 1)
plt.legend()
plt.show()

#X = np.c_[np.ones((X.shape[0], 1)), X]

## logistic regression
def f_sigmoid(x,A,w,x0,S):
    y = A/(1 + np.exp(-(x-x0)/w))+S
    return y

epsilon = 1e-5
def obj(x,y,A,w,x0,S):
    loss = np.mean(-y*(np.log(f_sigmoid(x,A,w,x0,S)+epsilon)) + (1-y)*np.log(1-f_sigmoid(x,A,w,x0,S)+epsilon))
    return loss

# def gradient(x, y,a,b):
#     m = X.shape[0]
#     h = f_sigmoid(x,a,b)
#     return (1/m) * np.dot(X.T, (h-y))

#res = minimize(lambda coeffs: obj(X_train_minmax, y_train, *coeffs), x0=,method = 'TNC')
#res = minimize(obj,args = (X_train_minmax, y_train), x0=np.array([1.3, 0.7, 0.8, 1.9]),method = 'TNC')

NFIT = 4
po=np.random.uniform(0.2,0.8,size=NFIT)
 #TRAIN MODEL USING SCIPY OPTIMIZER
from scipy.optimize import minimize
res = minimize(lambda coeffs:obj(X_train_minmax, y_train, *coeffs), po, method = 'TNC',tol=1e-15)
popt=res.x
print("OPTIMAL PARAM:",popt)

popt=res.x
print("OPTIMAL PARAM:",popt)

a = res.x[0]
b = res.x[1]
c = res.x[2]
d = res.x[3]
fig, ax = plt.subplots()
ax.plot(X_train_minmax,y_train,'o',X_train_minmax,f_sigmoid(X_train_minmax,a,b,c,d),'-')
plt.show()


iterations=[]; loss_train=[];  loss_val=[]

iteration=0
def loss(p):
      global iteration,iterations,loss_train,loss_val
       ....
      loss_train.append(training_loss)
      loss_val.append(validation_loss)
      iterations.append(iteration)

       iteration+=1


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import minimize


# ax.plot(xe, ye, '-', label='Ground-Truth')
# ax.plot(xe, model(xe, popt), 'r-', label="Model")



# x = np.array([139, ...])
# y = np.array([151, ...])

# Define the Model
def f(x, a, b): return a * x + b

# The objective Function to minimize (least-squares regression)
def obj(x, y, a, b):
    return np.sum((y - a*x+b)**2)

# define the bounds -infty < a < infty,  b <= 0
bounds = [(None, None), (None, 0)]

# res.x contains your coefficients
res = minimize(lambda coeffs: obj(X, y, *coeffs), x0=np.zeros(2), bounds=bounds)


#INITIAL GUESS
xo=xmax #
#xo=np.random.uniform(xmin,xmax)
print("INITIAL GUESS: xo=",xo, " f(xo)=",f(xo))
res = minimize(f1, xo, method='Nelder-Mead', tol=1e-5)
popt=res.x
print("OPTIMAL PARAM:",popt)



#FUNCTION TO OPTIMZE
def f(x):
	out=x**2.0
	# out=(x+10*np.sin(x))**2.0
	return out

#PLOT
#DEFINE X DATA FOR PLOTTING
N=1000; xmin=-20; xmax=120
X = np.linspace(xmin,xmax,N)

plt.figure() #INITIALIZE FIGURE
FS=18   #FONT SIZE
plt.xlabel('x', fontsize=FS)
plt.ylabel('f(x)', fontsize=FS)
plt.plot(X,y,'-')

num_func_eval=0
def f1(x):
	global num_func_eval
	out=f(x)
	num_func_eval+=1
	if(num_func_eval%10==0):
		print(num_func_eval,x,out)
	plt.plot(x,f(x),'ro')
	plt.pause(0.11)

	return out

#INITIAL GUESS
xo=xmax #
#xo=np.random.uniform(xmin,xmax)
print("INITIAL GUESS: xo=",xo, " f(xo)=",f(xo))
res = minimize(f1, xo, method='Nelder-Mead', tol=1e-5)
popt=res.x
print("OPTIMAL PARAM:",popt)

plt.show()



