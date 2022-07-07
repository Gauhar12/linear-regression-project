#!/usr/bin/env python
# coding: utf-8

# Introduction
# feature_2 = θ0 + θ1 . feature_1

# In[1]:


from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[127]:


x = np.arange(-5,5)
y = 1 + 2 * x
y2 = 1 + 12 * x
plt.plot(x,y, 'r')
plt.plot(x,y2, 'b')
plt.grid()


# Write the linear hypothesis function.

# In[128]:


def h(x, theta):
    return np.dot(x, theta)


# In[130]:


theta = np.array([1,2])
new_column = np.ones((10,1))
x = x.reshape(10,1)
x = np.append(new_column, x, axis = 1)


# In[131]:


h(x, theta)


# Write the Mean Squared Error function between the predicted values and the labels.

# In[132]:


def mean_squared_error(y_predicted, y_label):
    return np.sum((y_predicted - y_label) ** 2)/len(y_label)


# Closed-Form Solution

# Write a class LeastSquareRegression to calculate the θ feature weights and make predictions.

# In[133]:


class LeastSquaresRegression():
    def __init__(self,):
        self.theta_ = None  
        
    def fit(self, X, y):
        p1 = np.dot(X.T, X)
        p2 = np.dot(X.T, y)
        my_inv = np.linalg.inv(p1)
        self.theta_ = np.dot(my_inv, p2)
        # Calculates theta that minimizes the MSE and updates self.theta_
    
        
    def predict(self, X):
        return h(X, self.theta_)
        # Make predictions for data X, i.e output y = h(X) (See equation in Introduction)


# In[134]:


xx = 4 * np.random.rand(100, 1)
y = 10 + 2 * xx + np.random.randn(100, 1)


# Plot these points to get a feel of the distribution.

# In[140]:




plt.scatter(xx,y)

plt.plot(xx,h(X_new, model.theta_),color = 'yellow')


# Write a function which adds one to each instance

# In[136]:


def bias_column(X):
    new_col = np.ones((100,1))
    X = np.append(new_col, X, axis = 1)
    return X

X_new = bias_column(xx)

print(xx[:5])
print(" ---- ")
print(X_new[:5])


# In[71]:


bias_column(xx)


# In[137]:


#Calculate the weights with the LeastSquaresRegression class
model = LeastSquaresRegression()
model.fit(X_new, y)
model.theta_


# Are the values consistent with the generating equation (i.e 10 and 2) ?

# In[138]:


h(X_new, model.theta_)


# Gradient Descent

# In[141]:


class GradientDescentOptimizer():

    def __init__(self, f, fprime, start, learning_rate = 0.1):
        self.f_      = f                       # The function
        self.fprime_ = fprime                  # The gradient of f
        self.current_ = start                  # The current point being evaluated
        self.learning_rate_ = learning_rate    # Does this need a comment ?

        # Save history as attributes
        self.history_ = start
    
    def step(self):
        # Take a gradient descent step
        # 1. Compute the new value and update selt.current_
        self.current_ = self.current_ - self.learning_rate_ * self.fprime_(self.current_)
        # 2. Append the new value to history
        self.history_ = np.append(self.history_, self.current_, axis = 1)
        # Does not return anything

        
    def optimize(self, iterations = 100):
        # Use the gradient descent to get closer to the minimum:
        # For each iteration, take a gradient step
        iters = 0
        while iters > iterations:
            self.step()            
            iters += 1

            
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_)))


# Write the f function f(x) = 3 + (x - (2  6)T)T · (x - (2  6)T).

# In[1]:


def f(x):
    a = np.array([[2], [6]])
    return 3 + np.dot((x-a).T,(x-a))


# Write the fprime function

# In[2]:


def fprime(x):
    a = np.array([[2], [6]])
    return 2 * (x-a)


# Use the the gradient descent optimizer to try to find the best theta value

# In[3]:


grad = GradientDescentOptimizer(f, fprime, np.random.normal(size=(2,1)), 0.1)
grad.optimize(100)
x_iterations = grad.print_result()


# In[4]:


plt.plot(x_iterations[0,:])
plt.plot(x_iterations[1,:])


# In[ ]:




