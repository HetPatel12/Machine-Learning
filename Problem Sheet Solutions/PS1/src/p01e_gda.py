import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel
from p01b_logreg import LogisticRegression


def main(train_path, eval_path, pred_path, box_cox=True):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model1 = GDA()
    model1.fit(x_train, y_train)
    
    x_train_, y_train_ = util.load_dataset(train_path, add_intercept=True)
    
    model2 = LogisticRegression()
    model2.fit(x_train_, y_train_)
    
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    x_train_plt, y_train_plt = util.load_dataset(train_path, add_intercept=False) # to plot
    
    if "ds1" in train_path:
        best_X = None
        if box_cox:
            print("-"*20, end='')
            print("Running BoxCox validation", end='')
            print("-"*20)
            box_cox_vals = [0.5, 0.75, 0.8, 1]
            best_acc = -float('inf')
            best_model = None
            for lambda_ in box_cox_vals:
                X = np.sign(x_train) * (np.abs(x_train)) ** (lambda_)
                model = GDA()
                model.fit(X, y_train)
                y_pred1 = model.predict(x_val)
                acc = 100*np.sum((y_pred1>0.5)==y_val)/len(y_val)
                if acc>best_acc:
                    best_X = X
                    print(f"Accuracy={acc}% for lambda={lambda_}")
                    best_acc = acc
                    best_model = model
        else:
            best_model = model1
            
        plt.plot(best_X[:, 0], best_X[:, 1], 'ro')
        plt.savefig('output/p01e_boxcox.png')
        plt.close('all')
        util.plot(x_train_plt, y_train_plt, model2.theta, save_path='output/p01b_ds1.png')
        util.plot(x_train_plt, y_train_plt, best_model.theta, save_path='output/p01e_ds1.png')
        y_pred1 = best_model.predict(x_val)
        y_pred2 = model2.predict(x_val)
        print("-"*20, end='')
        print("dataset1", end='')
        print("-"*20)
        print(f"Accuracy of GDA on validation set is: {100*np.sum((y_pred1>0.5)==y_val)/len(y_val)}%")
        print(f"accuracy of Logistic Regression on validation set is: {100*np.sum((y_pred2>0.5)==y_val)/len(y_val)}%")
        
        
    else:
        util.plot(x_train_plt, y_train_plt, model2.theta, save_path='output/p01b_ds2.png')
        util.plot(x_train_plt, y_train_plt, model1.theta, save_path='output/p01e_ds2.png')
        y_pred1 = model1.predict(x_val)
        y_pred2 = model2.predict(x_val)
        print("-"*20, end='')
        print("dataset2", end='')
        print("-"*20)
        print(f"Accuracy of GDA on validation set is: {100*np.sum((y_pred1>0.5)==y_val)/len(y_val)}%")
        print(f"accuracy of Logistic Regression on validation set is: {100*np.sum((y_pred2>0.5)==y_val)/len(y_val)}%")
        
    np.savetxt(pred_path, y_pred1>0.5, fmt="%d")
    
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        phi = np.sum(y)/len(y)
        u1 = np.sum(x[y==1], axis=0)/np.sum(y)
        u0 = np.sum(x[y==0], axis=0)/np.sum(1-y)
        sigma = ((x[y==1]-u1).T)@(x[y==1]-u1) + ((x[y==0]-u0).T)@(x[y==0]-u0)
        # self.theta = {"phi":phi,
        #               "u1":u1,
        #               "u0":u0,
        #               "sigma":sigma}
        
        theta = np.zeros(n+1)
        sigma_inv = np.linalg.inv(sigma)
                
        theta[0] = 0.5*(- u1.T.dot(sigma_inv).dot(u1) + u0.T.dot(sigma_inv).dot(u0) + np.log((1-phi)/phi))
        theta[1:] = sigma_inv.dot(u1-u0)
        
        self.theta = theta
        
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # u1 = self.theta["u1"]
        # u0 = self.theta["u0"]
        # phi = self.theta["phi"]
        # sigma = self.theta["sigma"]
        
        # py1 = (x-u1).T.dot(np.linalg.inv(sigma)).dot(x-u1)*phi
        # py0 = (x-u0).T.dot(np.linalg.inv(sigma)).dot(x-u0)*phi
        
        #y_pred = py1>py0
        
        y_pred = 1/(1 + np.exp(-x.dot(self.theta)))
        
        return y_pred
        # *** END CODE HERE