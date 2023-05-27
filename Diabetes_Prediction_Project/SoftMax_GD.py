import math
import numpy as np

class SoftMax_alg:
    import math
    import numpy as np
    def __init__(self, name):
        self.name = name

    def softmax_score_k(self, parameter_matrix, instance):
        
        param_vectors = parameter_matrix  
        sm_scores = np.dot(instance, param_vectors.T)  # Compute the scores using matrix multiplication (1 * 4) dot (4 * 3)
        
        return sm_scores


    def softmax(self, sm_scores):  #sm_scores is 1*3 array
        
        sm_scores = np.array(sm_scores)  # Ensure that sm_scores is a NumPy array 
        exp_scores = np.exp(sm_scores - np.max(sm_scores))  # Subtracting max for numerical stability  
        p_hat_vector = exp_scores / np.sum(exp_scores)  # Normalize the scores to get probabilities w/ scalar multiplication
        
        return p_hat_vector

    def p_hat_matrix_maker(self, X_train, parameter_matrix):
        
        m, n = X_train.shape
        k = 2 #for binary
        # Initialize p_hat_matrix with the correct shape
        p_hat_matrix = np.empty((0, k))

        for i in range(len(X_train)):
            score_list = self.softmax_score_k(parameter_matrix, X_train[i])
            p_hat_vector = self.softmax(score_list)

            # Reshape p_hat_vector and concatenate it to p_hat_matrix
            p_hat_vector = np.array(p_hat_vector).reshape(1, -1)
            p_hat_matrix = np.concatenate((p_hat_matrix, p_hat_vector), axis=0)
        
        return p_hat_matrix

    def predict(self, parameter_matrix, X):
        
        predictions = []
        
        for instance in X:                                           #apply softmax pipeline to all instances in a given X set
            sm_scores = self.softmax_score_k(parameter_matrix, instance)
            p_hat = self.softmax(sm_scores)
            class_prediction = np.argmax(p_hat)
            predictions.append(class_prediction)
        
        return np.array(predictions)


    def cross_entropy_cost(self, X_features, y_target, p_hat_matrix):
        
        m = len(X_features)  # Get the number of instances from the length of X_features
        k = p_hat_matrix.shape[1]  # Get the number of classes from the second dimension of p_hat_matrix

        # One-hot encoding for target values
        one_hot_target = np.zeros((m, k))   #np array filled with zeroes in the shape (m, k)
        one_hot_target[np.arange(m), y_target] = 1  #np.arrange(m) creates array w/ elements correlating to isntance row index
                                                    #y_target will column index we want to set to 1
                                                    #alltogether this statement sets the index that y_target is to 1 for every row

        # Compute the cross-entropy cost function
        cross_entropy = (-1 / m) * np.sum(one_hot_target * np.log(p_hat_matrix))

        return cross_entropy

    def cross_entropy_gradient_vector(self, X_features, p_hat_matrix, y_target):
        
        m, n = X_features.shape  # Get the number of instances (m) and number of features (n) from X_features
        k = p_hat_matrix.shape[1]  # Get the number of classes from the second dimension of p_hat_matrix

        # One-hot encoding for target values
        one_hot_target = np.zeros((m, k))
        one_hot_target[np.arange(m), y_target] = 1

        # Calculate the gradient matrix using NumPy operations
        gradient_matrix = np.dot((p_hat_matrix - one_hot_target).T, X_features) / m

        return gradient_matrix

    def gradient_descent_step(self, parameter_matrix, learning_rate, gradient_matrix):
        next_step = parameter_matrix - (learning_rate * gradient_matrix) #next step is the revised param matrix
        
        return next_step

    def GD_main(self, X_train, y_target, parameter_matrix, epochs, learning_rate):
        '''
        GD_main(self, X_train, y_target, parameter_matrix, epochs, learning_rate):
        '''
        parameter_matrix = parameter_matrix
        
        for epoch in range(epochs):
            p_hat_matrix = self.p_hat_matrix_maker(X_train, parameter_matrix)  #get probabilities

            cross_entropy = self.cross_entropy_cost(X_train, y_target, p_hat_matrix)  #get cost

            gradient_matrix = self.cross_entropy_gradient_vector(X_train, p_hat_matrix, y_target)  #compute gradient

            next_step = self.gradient_descent_step(parameter_matrix, learning_rate, gradient_matrix)  #compute new params
            
            parameter_matrix = next_step #assign new params
            
            print(f'epoch: {epoch} ---- cost: {cross_entropy}')  
            
        return parameter_matrix