import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    # Fit a Poisson Regression model
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    
    y_pred = model.predict(x_val)
    plt.plot(y_pred, y_val, 'b+')
    plt.savefig('output/p03d.png')
    plt.close('all')

    
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        self.theta = np.random.randn(n)
        i = 0
        
        #for i in range(self.max_iter):
        while True:
            
            hx = np.exp(x.dot(self.theta))
            old_theta = self.theta
            
            grad_theta = (x.T).dot(y - hx)/m
            
            loss = np.sum(y - hx)/m
            self.theta = self.theta + self.step_size*grad_theta
            
            if self.verbose and (i+1)%(self.max_iter/10)==0:
                #print(grad_theta)
                print(f"Iteration {i+1}, Loss : {loss:.4f}")
            
            if np.linalg.norm(self.theta - old_theta, ord=1) < self.eps:
                print(f"Final Iteration {i+1}, Loss : {loss:.4f}")
                break
            
            i += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
