import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        
        while True:
            
            old_theta = self.theta
            
            hx = 1/(1 + np.exp(-x.dot(self.theta)))
            H = (x.T * hx * (hx - 1)).dot(x)/m
            grad_l = x.T.dot(y - hx)/m
            
            self.theta = self.theta - np.linalg.inv(H).dot(grad_l)
            
            if(np.linalg.norm(self.theta-old_theta, ord=1) < self.eps):
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        hx = x.dot(self.theta)
        hx = 1/(1 + np.exp(-hx))
        
        return hx
        # *** END CODE HERE ***
