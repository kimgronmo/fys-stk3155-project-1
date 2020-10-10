
# 
# print doc


# imports functions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import to check code
import sklearn.linear_model as skl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# imports linear regression from scikitlearn to 
# check my code against
from sklearn.metrics import mean_squared_error

# imports resample from sklearn
from sklearn.utils import resample

# import for cross validation
# should use 5-10 splits
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# imports Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# imports functions for real world data
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

# Where figures and data files are saved..
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)
def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)
def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)
def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')
    
# defines some statistical functions
# make proper note notation or function descriptions for Python
# @@@??
def R2(y_data, y_model):
    return 1 - ( (np.sum((y_data - y_model) ** 2)) / (np.sum((y_data - np.mean(y_data)) ** 2)) )
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# own code for svd algorithm pensum side 25
# skriv om litt og gjør egne kommentarer
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    # print('test U')
    # print( (np.transpose(U) @ U - U @np.transpose(U)))
    # print('test VT')
    # print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))
    #print(U)
    #print(s)
    #print(VT)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
        
    # inv or pinv here????
    # what is the difference?
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.pinv(D)
    return np.matmul(V,np.matmul(invD,UT))

# defines some basic functions, note where they are from.
# given by the projects assignment
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# creates the design matrix, from lecture materials p20
# make sure to comment the code properly
def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x) # what are we doing here?
        y = np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X
    
# Start by defining the different reggression functions:
# OLS, Ridge, Lasso

# Ordinary Linear Regression
# returns beta values and checks against sklearn for errors
def OLS(xtrain,ytrain):
    # Testing my regression versus sklearn version
    # svd inversion ols regression
    OLSbeta_svd = SVDinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain
    return OLSbeta_svd

# Ridge Regression
# returns beta values  
def RidgeManual(xtrain,lmb,identity,ytrain):
    Ridgebeta = SVDinv((xtrain.T @ xtrain) + lmb*identity) @ xtrain.T @ ytrain
    #print("Ridgebeta in function size is: ",Ridgebeta.size)
    return Ridgebeta
    
# Generates and prints the Confidence Intervals for the betavalues
def betaConfidenceIntervals(X_train,betaValues,y_train,ytilde):

    print("\nCalculating and printing the Confidence Intervals for Beta:")

    # Calculating sample variance
    # N-p-1 for unbiased estimator, (3.8) from Hastie..
    # am I using correct p?
    sampleVar = np.sum((y_train-ytilde)**2)/(ytilde.size- betaValues.size- 1)
    #print(sampleVar)

    # the variances are the diagonal elements
    betaVariance = np.diag(SVDinv(X_train.T @ X_train))
    sDev = np.sqrt(sampleVar)
    
    # For a 95% confidence interval the z value is 1.96
    zvalue = 1.96
    confidence = sDev*zvalue*np.sqrt(betaVariance)

    print("Lower bound:    Beta Values:    Upper bound:")    
    for i in range(betaValues.size):
        lower = betaValues[i] - confidence[i]
        upper = betaValues[i] + confidence[i]
        print(lower, betaValues[i], upper)

def figure211():
    # makes figure 2.11 from Hastie..
    print("...this might take a wile... change maxPoly to speed up..")
    #print("#",end='')
    #print("I'm running this function")
    maxPoly = 50
    numPoly = np.zeros(maxPoly)
    mse_train211=[]
    mse_test211=[]
    for i in range(maxPoly):
        # Trying to print a progress bar...
        #print("#",end=''))
        numPoly[i] = i
        X = create_X(x, y, n=i)

        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)        

        betaValues = OLS(X_train,y_train)
        ytilde = X_train @ betaValues
        ypredict = X_test @ betaValues 
        mse_train211.append(MSE(y_train,ytilde))
        mse_test211.append(MSE(y_test,ypredict))
        #a progress bar

    #print(mse_train211)
    #print(mse_test211)
    # Generates plot and saves it in folder

    plt.figure(1)
    plt.title("Test and training error as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"mean square error", fontsize=10)
    plt.plot(numPoly, mse_train211, label = "MSE training")
    plt.plot(numPoly, mse_test211, label = "MSE test data")
    plt.legend([r"mse from training data", r"mse from test data"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'figure211 from Hastie sample 1000.png'), transparent=True, bbox_inches='tight')
    #plt.show()    

# bootstrap
def bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,regression,lambda1):
    if(regression == "OLS"):
        #print("We are doing OLS regression")
        for i in range(n_bootstraps):
            x_, y_ = resample(X_train,y_train)
            #print("trying first loop")
            # fits the data on same test data each time
            OLSbeta_svd_scaled = SVDinv(x_.T @ x_) @ x_.T @ y_
            y_pred[:,i] = (X_test @ OLSbeta_svd_scaled).ravel()
            y_pred[:,i] = (y_pred[:,i].T).ravel()
    if(regression == "Ridge"):
        #print("We are doing Ridge regression")
        sizeofX = X_train.shape
        tempmatrix = (X_train.T @ X_train)
        Identity = np.eye(tempmatrix.shape[0])
        #print("got this far??")
        #print(I.shape)
        
        for i in range(n_bootstraps):
            #print("bootstrap: ",i," of ",n_bootstraps)
            x_, y_ = resample(X_train,y_train)
            sizeofX = x_.shape
            tempmatrix = (x_.T @ x_)
            I = np.eye(tempmatrix.shape[0])
            #print("got this far??")
            #print(I.shape)
            #print("lambda1 is :",lambda1)
            
            Ridgebeta = RidgeManual(x_,lambda1,I,y_)
            #Ridgebeta = SVDinv(x_.T @ x_ + lambda1*I) @ x_.T @ y_
            #print("ridgebetas: ",Ridgebeta)
            #print("X_test size is: ",X_test.size)
            #print("Ridgebeta size is: ",Ridgebeta.size)
            y_pred[:,i] = (X_test @ Ridgebeta).ravel()
            y_pred[:,i] = (y_pred[:,i].T).ravel()
           

    if(regression == "LASSO"):
        print("We are doing LASSO regression")

    return y_pred
    
# Studying bias and variance trade-off as a function of model complexity
def biasVariance(regression):
    for polydegree in range(1, maxPolyDegree):
        polynomial[polydegree] = polydegree
        X = create_X(x, y, n=polydegree)
        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # something wrong with train test split when splitting..
        y_test.shape = (y_test.shape[0], 1)
        #y_train.shape = (y_train.shape[0], 1)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)

        # force yshape??
        y_pred = np.empty((y_test.shape[0], n_bootstraps))    

        #Lets do the bootstrap analysis
        if(regression=="OLS"):
            y_pred = bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,"OLS",0)
        if(regression=="Ridge"):
            print("Starting bootstrap for polynomial: ",polydegree)
            # needs y_pred for each lambda
            counter1=0
            y_pred_lambda = []
            mse_valuesRidge=np.zeros(nlambdas)
            for lmb in lambdas:
                y_pred = bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,"Ridge",lmb)
                y_pred_lambda.append(y_pred)
                
                # calculating mse dependent on lambda
                mse_valuesRidge[counter1]= MSE(y_test,y_pred)
                counter1 +=1
                
            # use this to determine best fit for ridge?? for lambda and polynomial.
            # print best fit Ridge
            print("MSE values for Ridge regression with bootstrap")
            print(mse_valuesRidge)
            #print(y_pred_lambda.size)

        # Computing values for each polynomial
        polynomial[polydegree] = polydegree
        error[polydegree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[polydegree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[polydegree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )
        
        if(regression=="OLS"):
            print('Polynomial degree:', polydegree)
            print('Error:', error[polydegree])
            print('Bias^2:', bias[polydegree])
            print('Var:', variance[polydegree])
            print('{} >= {} + {} = {}'.format(error[polydegree], bias[polydegree], variance[polydegree], bias[polydegree]+variance[polydegree]))
    
    if(regression=="OLS"):
    # Plotting data for b)
        plt.figure(2)
        plt.title("Error (MSE) as a function of model complexity", fontsize = 10)    
        plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
        plt.ylabel(r"mean square error", fontsize=10)
        plt.plot(polynomial, error, label = "Error (MSE)")
        plt.legend([r"mse from training data"], fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'part b MSE vs poly sample 1000.png'), transparent=True, bbox_inches='tight')
    
        plt.figure(3)
        plt.title("Bias as a function of model complexity", fontsize = 10)    
        plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
        plt.ylabel(r"Bias", fontsize=10)
        plt.plot(polynomial, bias, label = "Bias")
        plt.legend([r"mse from training data"], fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'part b Bias vs poly sample 1000.png'), transparent=True, bbox_inches='tight')
    
        plt.figure(4)
        plt.title("Variance as a function of model complexity", fontsize = 10)    
        plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
        plt.ylabel(r"Variance", fontsize=10)
        plt.plot(polynomial, variance, label = "Variance")
        plt.legend([r"mse from training data"], fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'part b Variance vs poly sample 1000.png'), transparent=True, bbox_inches='tight')

# Cross Validation
def crossValidation(polynomial,regression):
    #  Cross-validation code
    print("Doing cross-validation for polynomial: ",polynomial)
    n = polynomial
    X = create_X(x, y, n=n)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

    # something wrong with train test split when splitting..
    y_test.shape = (y_test.shape[0], 1)

    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    # lambda values
    #nlambdas = 500
    nlambdas = 10
    lambdas = np.logspace(-6,5, nlambdas)

    # Initialize a Kfold instance
    k = 8
    kfold = KFold(n_splits = k)

    # Perform the cross-validation to estimate MSE
    scores_KFold = np.zeros((nlambdas, k))

    # Do ridge and cross validation at the same time??
    # c) and d)

    mse_values = np.zeros(k)

    counter = 0
    for train_inds, test_inds in kfold.split(X,z):
        #j=0
        X_train = X[train_inds]
        ytrain = z[train_inds]

        X_test = X[test_inds]
        ytest = z[test_inds]

        # scaling the data as usual
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)

        # checking to see if I have to reshape y
        y_test.shape = (y_test.shape[0], 1)

        if(regression == "OLS"):
            OLSbeta_svd_scaled = SVDinv(X_train.T @ X_train) @ X_train.T @ ytrain
            y_pred = (X_test @ OLSbeta_svd_scaled).ravel()
        if(regression == "Ridge"):
            #print("test")
            # for each lambda make an ypred??
            as1=1 # indentation error

        # calculate mse for each split, then sum and average. Compare
        # with earlier code
        if(regression == "OLS"):
            mse_values[counter]=MSE(ytest,y_pred)
            #print(mse_values[counter])


        # testing if fit is correct
        #print(R2(ytest,y_pred))
        counter += 1
    #print(mse_values.size)

    if (regression == "OLS"):
        print("Average MSE calculated from OLS cross-validation: ",np.mean(mse_values))

def LassoRegression(X_train,X_test,lambdas,y_train,y_test,fromSource):
    #print("in this function")
    MSEPredictLasso = np.zeros(nlambdas)
    MSEPredictRidge = np.zeros(nlambdas)
    lambdas = np.logspace(-4, 0, nlambdas)
    #mse_min = 100.0
    for i in range(nlambdas):

        # for ridge regression
        sizeofX = X_train.shape
        tempmatrix = (X_train.T @ X_train)
        Identity = np.eye(tempmatrix.shape[0])
        Ridgebeta = RidgeManual(X_train,lambdas[i],Identity,y_train)
        
        y_predict = (X_test @ Ridgebeta).ravel() 
        lmb = lambdas[i]
        # add ridge
        #clf_ridge = skl.Ridge(alpha=lmb).fit(X_train, y_train)
        clf_lasso = skl.Lasso(alpha=lmb).fit(X_train, y_train)
        #yridge = clf_ridge.predict(X_test)
        ylasso = clf_lasso.predict(X_test)
        MSEPredictLasso[i] = MSE(y_test,ylasso)
        MSEPredictRidge[i] = MSE(y_test,y_predict)
    
    # Finds the minimum values calculated
    if (fromSource == "summary"):
        mse_min_LASSO = 100.0
        mse_minRidge = 100.0
        print("Minimum values for polynomial: 5 are: ")
        result = np.where(MSEPredictLasso == np.amin(MSEPredictLasso))
        #print(result)
        print('Minimum MSE values for LASSO :', MSEPredictLasso[result[0]], "for lambda: ",lambdas[result[0]])
        result = np.where(MSEPredictRidge == np.amin(MSEPredictRidge))
        print('Minimum MSE values for Ridge :', MSEPredictRidge[result[0]], "for lambda: ",lambdas[result[0]])        
 
        
    #then plot the results
    plt.figure()
    plt.plot(np.log10(lambdas), MSEPredictRidge, 'r--', label = 'MSE Ridge Test')
    plt.plot(np.log10(lambdas), MSEPredictLasso, 'g--', label = 'MSE Lasso Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.legend()
    
    if (fromSource == "realworld"):
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles',
        'MSE LASSO vs Ridge terrain data.png'), transparent=True, bbox_inches='tight')
    if (fromSource == "franke"):    
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles',
        'MSE LASSO vs Ridge franke function.png'), transparent=True, bbox_inches='tight')
    #plt.show()
def summaryRegression(X_train,X_test,lambdas,y_train,ytest):
    print("\nSummary of regression\n")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"summary")            
    betaValues = OLS(X_train,y_train)
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    print("Minimum MSE values for OLS : ",MSE(y_test,ypredict)) 
 

# Using a seed to ensure that the random numbers are the same everytime we run
# the code. Useful to debug and check our code.
np.random.seed(3155)

# The degree of the polynomial (number of features) is given by
n = 5
# the number of datapoints
N = 1000
# the highest number of polynomials
maxPolyDegree = 10
# number of bootstraps
n_bootstraps = 100

# lambda values
#nlambdas = 500
nlambdas = 10
lambdas = np.logspace(-6,5, nlambdas)



x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)

# Remember to add noise to function 
z = FrankeFunction(x, y) + 0.01*np.random.rand(N)

X = create_X(x, y, n=n)

# split in training and test data
# assumes 75% of data is training and 25% is test data
X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

# scaling the data
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

error = np.zeros(maxPolyDegree)
bias = np.zeros(maxPolyDegree)
variance = np.zeros(maxPolyDegree)
polynomial = np.zeros(maxPolyDegree)


if __name__ == '__main__':
    print("---------------------")
    print("Running main function")
    print("---------------------")
    print("\n""\n")

    print("Starting Project 1: part a")
    print("\n")
    # data has already been scaled..
    betaValues = OLS(X_train,y_train)
    #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape,z.shape)
    #print("betavalues for real world data size is :",betaValues.size)    
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: ",MSE(y_test,ypredict))
    print("My R2 score is: ",R2(y_test,ypredict))
    
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling {:.4f}".format(clf.score(X_test,y_test)))
    
    # Calculates and prints the beta confidence intervals
    betaConfidenceIntervals(X_train,betaValues,y_train,ytilde)
    print("\n""\n")

    print("Starting Project 1: part b")

    # Generating figure 2.11 from Hastie..
    print("\nGenerating figure 2.11 from Hastie..")
    figure211()
    
    print("\nComputing bias and variance trade-off as a function of model complexity\n")
    biasVariance("OLS")
    
    print("\nStarting Project 1: part c\n")
    
    print("\nStarting Cross-validation\n")
    crossValidation(5,"OLS")
    
    print("\nStarting Project 1: part d\n")
    biasVariance("Ridge")
    crossValidation(5,"Ridge")
 
    print("LASSO regression")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"franke")

    # Summary of regression results
    summaryRegression(X_train,X_test,lambdas,y_train,y_test)
    

    print("\n\n###########################################################")
    print("###########################################################")
    
    print("\nStarting Project 1: part f and g\n")
    #terrain1 = imread(data_path("SRTM_data_Norway_1.tif"))
    terrain1 = Image.open(data_path("SRTM_data_Norway_1.tif"), mode='r')
    terrain1.mode = 'I'    

    #image = Image.open(file, mode='r')
    #image.mode = 'I'
    
    # Problem making the design matrix when the image is not a square
    x = np.linspace(0, 1, int(terrain1.size[0]/5))
    #print("terrain size x is ",terrain1.size[0])
    y = np.linspace(0, 1, int(terrain1.size[0]/5))
    #print("terrain size y is ",terrain1.size[1])
    z_temp = np.array(terrain1)
    #print(z_temp)
    z_temp = z_temp - np.min(z_temp)
    z_temp = z_temp / np.max(z_temp)
    z = np.empty([int(terrain1.size[0]/5),int(terrain1.size[0]/5)])
    for i in range(int(terrain1.size[0]/5)):
        for j in range(int(terrain1.size[0]/5)):
            z[i,j]=z_temp[i,j]
            #er=1
    z = z.flatten()
    z=z[:y.size]
    #print("z size is: ",z.size)
    #print
    X = create_X(x, y, n=5)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

    # Checking to see if the matrices has correct shape
    #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape,z.shape)
    #print (y_test)
    
    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    error = np.zeros(maxPolyDegree)
    bias = np.zeros(maxPolyDegree)
    variance = np.zeros(maxPolyDegree)
    polynomial = np.zeros(maxPolyDegree)    
    
    # data has already been scaled..
    betaValues = OLS(X_train,y_train)
    #print("betavalues for real world data size is :",betaValues.size)
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: ",MSE(y_test,ypredict))
    print("My R2 score is: ",R2(y_test,ypredict))
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling {:.4f}".format(clf.score(X_test,y_test)))
    
    # Calculates and prints the beta confidence intervals
    betaConfidenceIntervals(X_train,betaValues,y_train,ytilde)
    print("\n""\n")
    print("Starting Project 1: part b")

    # Generating figure 2.11 from Hastie..
    print("\nGenerating figure 2.11 from Hastie..")
    figure211()
    
    print("\nComputing bias and variance trade-off as a function of model complexity\n")
    biasVariance("OLS")
    
    print("\nStarting Project 1: part c\n")
    
    print("\nStarting Cross-validation\n")
    crossValidation(5,"OLS")
    
    print("\nStarting Project 1: part d\n")
    biasVariance("Ridge")
    crossValidation(5,"Ridge")
 
    print("LASSO regression")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"realworld")

    #plotting terrain data
    plt.figure()
    plt.title("Terrain data Norway")
    plt.imshow(terrain1, cmap="gray")
    #plt.show()
 
    # Summary of regression results
    summaryRegression(X_train,X_test,lambdas,y_train,y_test) 
    
