import numpy as np
import math
def logistic(z):
    """
    The logistic function
    Input:
       z   numpy array (any shape)
    Output:
       p   numpy array with same shape as z, where p = logistic(z) entrywise
    """
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    p = np.full(z.shape, 0.5)
    for i in range(0,len(z)):
        p[i]=1/(1+math.e**(-z[i]))
    return p

def cost_function(X, y, theta):
    """
    Compute the cost function for a particular data set and hypothesis (weight vector)
    Inputs:
        X      data matrix (2d numpy array with shape m x n)
        y      label vector (1d numpy array -- length m)
        theta  parameter vector (1d numpy array -- length n)
    Output:
        cost   the value of the cost function (scalar)
    """
    m,n=X.shape
    
    # REPLACE CODE BELOW WITH CORRECT CODE
    cost = 0
    for i in range(0,m):
        temp=np.array([np.vdot(X[i],np.transpose(theta))])
        cost=((-y[i]*np.log(logistic(temp)))-(1-y[i])*(np.log(1-logistic(temp))))+cost
    return cost


def gradient(X,y, theta):
    m,n=X.shape
    inner=[0]*n
    for i in range(0,n):
        temp1=0
        for j in range(0,m):
            temp2=np.array([np.vdot(X[j],np.transpose(theta))])
            temp1+=(logistic(temp2)-y[j])*X[j][i]
        
        inner[i]=temp1
        
    return inner
        

def gradient_descent( X, y, alpha, iters, theta ):
    """
    Fit a logistic regression model by gradient descent.
    Inputs:
        X          data matrix (2d numpy array with shape m x n)
        y          label vector (1d numpy array -- length m)
        theta      initial parameter vector (1d numpy array -- length n)
        alpha      step size (scalar)
        iters      number of iterations (integer)
    Return (tuple):
        theta      learned parameter vector (1d numpy array -- length n)
        J_history  cost function in iteration (1d numpy array -- length iters)
    """

    list1=[]
    m,n = X.shape

    # For recording cost function value during gradient descent
    J_history = np.zeros(iters)

    for i in range(0, iters):

        # TODO: compute gradient (vectorized) and update theta
        
            
        #gets partial derivative with respect to theta_j
        list=np.array(gradient(X,y,theta))
       
        theta= theta-alpha*list[0]
        
        # Record cost function
        J_history[i] = cost_function(X, y, theta)
    
    list1=[theta,J_history]
    return list1