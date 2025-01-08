import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    MSE = np.sum((y_pred - y_val)**2)/(len(y_val))
    print(f"MSE is : {MSE:.4f}")
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    plt1 = plt.plot(x_train, y_train, 'bx') 
    plt2 = plt.plot(x_val, y_val, 'y+') 
    plt3 = plt.plot(x_val, y_pred, 'ro')
    plt.legend(handles = [plt1[1], plt2[1], plt3[1]],labels = ['Training data', 'Validation set data true labels', 'Validation set data predicted labels'])
    plt.savefig('output/p05b_underfitting.png')
    plt.show()
    plt.close('all')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        X = self.x
        Y = self.y
        tau = self.tau
        
        m, n = X.shape
        m_, n_ = x.shape
        y_pred = np.zeros(m_)
        
        for i in range(m_):
            W = (0.5)*np.diag(np.exp(-np.sum((x[i]-X)**2, axis=1)/(2 * tau**2))) #(0.5)*np.identity(m)*(np.exp(-((x[i]-X).dot((x[i]-X).T))/(2 * tau**2)))
            theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
            y_pred[i] = theta.T.dot(x[i])
            
        return y_pred
        
        # *** END CODE HERE ***
