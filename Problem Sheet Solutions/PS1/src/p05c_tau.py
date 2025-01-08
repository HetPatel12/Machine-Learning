import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    best_tau = tau_values[0]
    best_model = None
    best_MSE = float('inf')
    
    for tau in tau_values:
        
        model_test = LocallyWeightedLinearRegression(tau=tau)
        model_test.fit(x_train, y_train)
        y_pred = model_test.predict(x_val)
        
        plt1 = plt.plot(x_val, y_val, 'y+')
        plt2 = plt.plot(x_val, y_pred, 'ro')
        plt.legend(handles = [plt1[1], plt2[1]],labels = ['Validation set data true labels', 
                                                          'Validation set data predicted labels'])
        
        plt.title(f"tau = {tau}")
        plt.savefig(f"output/p05c_{tau}.png")
        plt.close('all')
        
        MSE = np.sum((y_val - y_pred)**2)/len(y_val)
        
        print(f"MSE = {MSE:.4f} at tau = {tau}", end='\n\n')
        
        if MSE < best_MSE:
            best_MSE = MSE
            best_tau = tau
            best_model = model_test
            
    print(f"best tau is: {best_tau}")
    
    # Fit a LWR model with the best tau value
    
    ### this model is best_model from above
    
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred_t = best_model.predict(x_test)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred_t)
    # Plot data
    plt1 = plt.plot(x_test, y_test, 'y+')
    plt2 = plt.plot(x_test, y_pred_t, 'ro')
    plt.legend(handles = [plt1[1], plt2[1]],labels = ['Test data true labels', 
                                                      'Test data predicted labels'])
    
    plt.savefig("output/p05c_test.png")
    plt.close('all')
    
    MSE = np.sum((y_pred_t - y_test)**2)/len(y_test)
    
    print(f"Test set MSE is: {MSE:.4f}")
    # *** END CODE HERE ***
