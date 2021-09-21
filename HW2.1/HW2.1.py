import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize

# USER PARAMETERS
IPLOT = True
INPUT_FILE = 'weight.json'
FILE_TYPE = "json"
DATA_KEYS = ['x', 'is_adult', 'y']
OPT_ALGO = 'BFGS'

# UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type = "logistic";
NFIT = 4;
xcol = 1;
ycol = 2;
# model_type="linear";   NFIT=2; xcol=1; ycol=2;
#model_type="logistic";   NFIT=4; xcol=2; ycol=0;

# READ FILE
with open(INPUT_FILE) as f:
    my_input = json.load(f)  # read into dictionary

# CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X = [];
for key in my_input.keys():
    if (key in DATA_KEYS): X.append(my_input[key])

# MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X = np.transpose(np.array(X))

# SELECT COLUMNS FOR TRAINING
x = X[:, xcol];
y = X[:, ycol]

# EXTRACT AGE<18
if (model_type == "linear"):
    y = y[x[:] < 18];
    x = x[x[:] < 18];

# COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN = np.mean(x);
XSTD = np.std(x)
YMEAN = np.mean(y);
YSTD = np.std(y)

# NORMALIZE
x = (x - XMEAN) / XSTD;
y = (y - YMEAN) / YSTD;

# PARTITION
f_train = 0.8;
f_val = 0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1 = int(f_train * x.shape[0]);
train_idx, val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt = x[train_idx];
yt = y[train_idx];
xv = x[val_idx];
yv = y[val_idx]


# MODEL
def model(x, p):
    #if (model_type == "linear"):   return p[0] * x + p[1]
    #if (model_type == "logistic"):
    return p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))


# SAVE HISTORY FOR PLOTTING AT THE END
iteration = 0;
iterations = [];
loss_train = [];
loss_val = []


# LOSS FUNCTION
def loss(p,idx):
    global iterations, loss_train, loss_val, iteration

    # samples input
    xt1 = xt[idx]
    yt1 = yt[idx]

    # TRAINING LOSS
    yp = model(xt1, p)  # model predictions for given parameterization p
    training_loss = (np.mean((yp - yt1) ** 2.0))  # MSE

    # VALIDATION LOSS
    yp = model(xv, p)  # model predictions for given parameterization p
    validation_loss = (np.mean((yp - yv) ** 2.0))  # MSE

    # WRITE TO SCREEN
    #if (iteration == 0):    print("iteration	training_loss	validation_loss")
    #if (iteration % 25 == 0): print(iteration, "	", training_loss, "	", validation_loss)

    # RECORD FOR PLOTING
    loss_train.append(training_loss);
    loss_val.append(validation_loss)
    iterations.append(iteration);
    iteration += 1

    return training_loss


def optimizer(loss, algo='GD', LR=0.01, method='batch'):
    # algo = 'GD' or 'GD+momentum'
    # method = 'batch', 'minibatch', or 'stochastic'

    # PARAM
    xmin = 0;
    xmax = 1;
    NDIM = 4
    xi = np.random.uniform(xmin, xmax, NDIM)  # INITIAL GUESS FOR OPTIMIZEER


    print("#--------Algo =",algo,"--------")

    # PARAM
    dx = 0.01  # STEP SIZE FOR FINITE DIFFERENCE
    LR = LR  # LEARNING RATE
    t = 0  # INITIAL ITERATION COUNTER
    tmax = 10000  # MAX NUMBER OF ITERATION
    tol = 10 ** (-30)  # EXIT AFTER CHANGE IN F IS LESS THAN THIS
    ICLIP = False

    print("INITAL GUESS: ", xi)

    rand_indices = np.random.permutation(xt.shape[0])
    if (method == 'batch'):
        i_per_epoch = 1
        batch_list = [rand_indices]

    elif(method == 'minibatch'):
        i_per_epoch = 2
        batch_size = 0.5
        CUT1 = int(batch_size * xt.shape[0])
        idx1, idx2 = rand_indices[:CUT1], rand_indices[CUT1:]
        batch_list = [idx1,idx2]
    elif(method == 'stochastic'):
        LR = 0.002
        tmax = 30000
        ICLIP = True
        i_per_epoch = xt.shape[0]
        batch_list = []
        for i in range(0,i_per_epoch):
            batch_list.append(rand_indices[i])


    momentum = 0.1
    past_velocity = np.zeros(NDIM)
    velocity = np.zeros(NDIM)

    while (t <= tmax):
        t = t+1


        for j in range(0,i_per_epoch):


            # NUMERICALLY COMPUTE GRADIENT
            df_dx = np.zeros(NDIM)
            for i in range(0, NDIM):
                dX = np.zeros(NDIM);
                dX[i] = dx
                xm1 = xi - dX  # print(xi,xm1,dX,dX.shape,xi.shape)
                df_dx[i] = (loss(xi,batch_list[j]) - loss(xm1,batch_list[j])) / dx

                if (ICLIP):
                    max_grad = 10
                    if (df_dx[i] > max_grad): df_dx[i] = max_grad
                    if (df_dx[i] < -max_grad): df_dx[i] = -max_grad

                if loss(xm1, batch_list[j]) > 0.01:
                    velocity[i] = past_velocity[i] * momentum + LR * df_dx[i]
            #print('training loss',loss(xm1,batch_list[j]))
            if (algo == 'GD'):
                xip1 = xi - LR * df_dx
            elif(algo =='GD+momentum'):
                xip1 = xi - LR * df_dx + velocity*momentum  # STEP
                past_velocity = velocity





        if (t % 100 == 0):
            df = np.mean(np.absolute(loss(xip1,batch_list[j]) - loss(xi,batch_list[j])))
            print(t, "	", xi, "	", "	", loss(xi,batch_list[j]) ,df,)

            if (df < tol):
                print("STOPPING CRITERION MET (STOPPING TRAINING)")
                break

        # UPDATE FOR NEXT ITERATION OF LOOP
        xi = xip1

    return xi



print("==========SHOWING MIMIBATCH, GD+momentum==========")
res = optimizer(loss,LR = 0.1,method = 'minibatch',algo = 'GD+momentum')

#res = optimizer(loss,LR = 0.1,algo = 'GD',method ='stochastic')


popt = res
print("OPTIMAL PARAM:", popt)

# PREDICTIONS
xm = np.array(sorted(xt))
yp = np.array(model(xm, res))


# UN-NORMALIZE
def unnorm_x(x):
    return XSTD * x + XMEAN


def unnorm_y(y):
    return YSTD * y + YMEAN

## plot the model
# fig, ax = plt.subplots()
# ax.plot(unnorm_x(xm),unnorm_y(yp),'o')
# plt.show()

#FUNCTION PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
	ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
	ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
	plt.xlabel('x', fontsize=18)
	plt.ylabel('y', fontsize=18)
	plt.legend()
	plt.show()

#PARITY PLOTS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(model(xt,popt), yt, 'o', label='Training set')
	ax.plot(model(xv,popt), yv, 'o', label='Validation set')
	plt.xlabel('y predicted', fontsize=18)
	plt.ylabel('y data', fontsize=18)
	plt.legend()
	plt.show()

#MONITOR TRAINING AND VALIDATION LOSS
if(IPLOT):
	fig, ax = plt.subplots()
	ax.plot(iterations, loss_train, 'o', label='Training loss')
	ax.plot(iterations, loss_val, 'o', label='Validation loss')
	plt.xlabel('optimizer iterations', fontsize=18)
	plt.ylabel('loss', fontsize=18)
	plt.legend()
	plt.show()