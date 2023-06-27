import numpy as np

class logistic_regression:
    def __init__(self, name):
        self.name = name

    def probability(self, param_vector, X):
        t = param_vector.dot(X.T) #(1, n) * (n, m) matrix multiplication results in (1, m) row vector of probailities 
        p_hat = 1 / (1 + np.exp(-t)) #Apply Logistic Function
        
        return p_hat  #(1, m)

    def model_prediction(self, m, prediction_threshold, p_hat):
        
        y_hat = np.zeros((1, m))
        
        positive = p_hat >= prediction_threshold #create boolean mask for pos predictions
        
        y_hat[positive] = 1
        
        return y_hat
        
    def log_loss_cost(self, y, p_hat, m):
        cost = (y * np.log(p_hat)) + ((1 - y) * np.log(1 - p_hat)) #cost for each instance
        J = (-1 / m) * np.sum(cost) # Compute (J(theta))
        
        return J

    def gradient_vector(self, p_hat, y, X):
        m, n = X.shape
        
        gradient = []
        
        for j in range(n):
            instance_error = (p_hat - y) * X[:, j]
            d_dtheta_j = (1/m) * np.sum(instance_error)  #scalar
            gradient.append(d_dtheta_j)
            
        return np.array(gradient)

    def gradient_descent(self, param_vector, X, y, epochs, learning_rate):
        
            m, n = X.shape
            
            param_history = []
            
            for epoch in range(epochs):
                p_hat = self.probability(param_vector, X)
                cost = self.log_loss_cost(y, p_hat, m)
                gradient = self.gradient_vector(p_hat, y, X)
                
                print(f"Epoch: {epoch}  -----  Cost: {cost}")
                param_history.append([epoch, param_vector])
                param_vector = param_vector - (learning_rate * gradient)
                    
            return param_vector, param_history
        