import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path, 
                                         label_col='t',
                                         add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, 
                                       label_col='t',
                                       add_intercept=True)
    
    model_c = LogisticRegression(eps=1e-5)
    model_c.fit(x_train, t_train)
    
    util.plot(x_test, t_test, model_c.theta, save_path='output/p02c.png')
    
    t_pred = model_c.predict(x_test)
    np.savetxt(pred_path_c, t_pred > 0.5, fmt='%d')
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path, 
                                         label_col='y',
                                         add_intercept=True)
    
    model_d = LogisticRegression(eps=1e-5)
    model_d.fit(x_train, y_train)
    
    util.plot(x_test, t_test, model_d.theta, save_path='output/p02d.png')
    
    y_pred = model_d.predict(x_test)
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_val, y_val = util.load_dataset(valid_path, 
                                     label_col='y',
                                     add_intercept=True)
    #y_pred2 = model_d.predict(x_val)
    alpha = np.sum(model_d.predict(x_val)*y_val)/np.sum(y_val)
    
    util.plot(x_test, t_test, model_d.theta, save_path='output/p02e.png', correction=alpha)
    
    t_pred = y_pred/alpha
    np.savetxt(pred_path_e, t_pred > 0.5, fmt='%d')
    # *** END CODER HERE
