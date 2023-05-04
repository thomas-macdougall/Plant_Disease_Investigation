import os
from sklearn.model_selection import GridSearchCV, KFold
from tensorflow.keras.utils import to_categorical
import argparse
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import skelm
from elm import ELM, ELMEstimatorWrapper
from Data.dark_subtraction import main as apply_dark_subtraction
import SuccessiveProjectionsAlgorithm as SPA
from analysis import analysis

# load the spectral data
def load_dataset(args):
    # ===============================
    # Load dataset
    # ===============================

    # load the data from the csv file into a numpy array, but not the first row
    X = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1, usecols=range(0, 154))

    n_classes = 4 #10
    num_input = 153 #28**2

    y = X[:, 0]
    X = X[:, 1:]

    # if feature_selection is True, then apply the feature selection algorithm
    if args.ftr_sel:
        spa = SPA.SuccessiveProjectionsAlgorithm()
        var_select = spa.select_features("./data.csv")
        # apply the feature selection algorithm
        X = X[:, var_select]
        num_input = len(var_select)
    
    return X, y, n_classes, num_input


def tune(args, x_train, t_train):

    # ===============================
    # Tune hyperparameters
    # ===============================
    if args.model == 'elm':
        elm = ELM(num_input, args.n_hidden_nodes, n_classes)

        estimator = ELMEstimatorWrapper(elm)
        param_grid = {
            'activation': ['tanh', 'sigmoid', 'relu', 'lin'],
            'loss': ['mean_absolute_error', 'mean_squared_error'],
            'n_hidden_nodes': [1024, 1536, 2048],
        }

    if args.model == 'skelm':
        estimator = skelm.ELMClassifier(
            classes=[0,1,2,3],
            n_neurons=args.n_hidden_nodes,
            )
        param_grid = {
            'alpha': [0.1, 0.01, 0.001, 0.0001],
            'n_neurons': [1024, 1536, 2048],
            'ufunc': ['tanh', 'sigm', 'relu', 'lin'],
            'pairwise_metric': ['euclidean', 'manhattan', 'cosine'],
        }

    if args.model == 'svm':
        # define the hyperparameter grid for SVM
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]
        }

        estimator = SVC(
            C=args.C,
            kernel='rbf',
            gamma='scale',
            decision_function_shape='ovr',
            random_state=42,
        )

    if args.model == 'logreg':
        # define the hyperparameter grid for logreg
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }

        estimator = LogisticRegression(
            penalty=args.penalty,
            C=args.C,
            multi_class='auto',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )


    if args.model == 'rf':
        # define the hyperparameter grid for random forest
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 'log2']
        }

        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            criterion='entropy',
            random_state=42,
            n_jobs=-1,
        )

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)

    # Perform the grid search
    grid_search.fit(x_train, np.argmax(t_train, axis=1))

    # Print the best hyperparameters and mean test score
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Mean test score: ", grid_search.best_score_)

    # write the best hyperparameters to a file
    with open('best_params.txt', 'a') as f:
        f.write('{}'.format(args.model) + str(grid_search.best_params_) + '\n')
    

# split the data into the format (x_train, t_train), (x_test, t_test)
def split_data(X,y, train_idx, test_idx, num_classes=4):

    # normalise the data
    X = X / np.max(X)

    x_train = X[train_idx]
    x_test = X[test_idx]
    t_train = y[train_idx]
    t_test = y[test_idx]

    t_train = to_categorical(t_train, num_classes).astype(np.float32)
    t_test = to_categorical(t_test, num_classes).astype(np.float32)

    return (x_train, t_train), (x_test, t_test)

def run(args, x_train, t_train, x_test, t_test, n_classes, num_input):

    if args.model == 'elm':
        elm = ELM(num_input, args.n_hidden_nodes, n_classes)

        elm.fit(x_train, t_train)
        y_pred = elm.predict(x_test)

        return np.argmax(y_pred, axis=1), np.argmax(t_test, axis=1)
    
    if args.model == 'skelm':

        elm = skelm.ELMClassifier(
            classes=[0,1,2,3],
            n_neurons=args.n_hidden_nodes,
            alpha=args.alpha,
            ufunc=args.ufunc,
            pairwise_metric=args.pairwise_metric
            )
        
        #print(np.argmax(t_train, axis=1))

        elm.fit(x_train, np.argmax(t_train, axis=1))

        y_pred = elm.predict(x_test)

        return y_pred, np.argmax(t_test, axis=1)

    if args.model == 'logreg':
        # ===============================
        # Instantiate Logistic Regression
        # ===============================
        model = LogisticRegression(
            penalty=args.penalty,
            C=args.C_log,
            multi_class='auto',
            solver=args.solver,
            max_iter=1000,
            random_state=42,
        )

        # ===============================
        # Training
        # ===============================
        model.fit(x_train, np.argmax(t_train, axis=1))

        # ===============================
        # Validation
        # ===============================
        y_pred = model.predict(x_test)

        return y_pred, np.argmax(t_test, axis=1)

    if args.model == 'rf':
        # ===============================
        # Instantiate Random Forest
        # ===============================
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            criterion='entropy',
            random_state=42,
            n_jobs=-1,
        )

        # ===============================
        # Training
        # ===============================
        model.fit(x_train, np.argmax(t_train, axis=1))

        # ===============================
        # Validation
        # ===============================
        y_pred = model.predict(x_test)

        return y_pred, np.argmax(t_test, axis=1)

    if args.model == 'svm':
        # ===============================
        # Instantiate SVM
        # ===============================
        model = SVC(
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma,
            decision_function_shape='ovr',
            random_state=42,
        )

        # ===============================
        # Training
        # ===============================
        model.fit(x_train, np.argmax(t_train, axis=1))

        # ===============================
        # Validation
        # ===============================
        y_pred = model.predict(x_test)

        return y_pred, np.argmax(t_test, axis=1)

# cross validation to evaluate the model
def cross_val(args, X, y, n_classes, num_input, num_folds=10, tuneing=False):
    # Initialize a K-Fold object with the specified number of folds
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Create empty lists to store the accuracy scores for each fold
    y_pred = []
    y_true = []

    # Loop over each fold
    for _, (train_idx, test_idx) in enumerate(kf.split(X)):

        (x_train, t_train), (x_test, t_test) = split_data(X, y, train_idx, test_idx, n_classes)

        if tuneing:
            tune(args, x_train, t_train)
        else:
            pred, true = run(args, x_train, t_train, x_test, t_test, n_classes, num_input)
        y_pred.append(pred)
        y_true.append(true)

    return y_pred, y_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--n_hidden_nodes', type=int, default=1536)
    parser.add_argument('--loss', choices=['mean_squared_error', 'mean_absolute_error'],default='mean_squared_error',)
    parser.add_argument('--activation',choices=['sigmoid', 'identity'],default='sigmoid',)
    parser.add_argument('--ftr_sel', action="store_true")
    parser.add_argument('--model', choices=['elm', 'logreg', 'rf', 'svm'], default='elm')
    parser.add_argument('--data', choices=['Dar5', 'CW14'], default='Dar5')
    parser.add_argument('--penalty', choices=['l1', 'l2'], default='l1')
    parser.add_argument('--C', type=float, default=0.1)
    parser.add_argument('--C_log', type=float, default=100)
    parser.add_argument('--solver', choices=['lbfgs', 'liblinear', 'sag', 'saga'], default='liblinear')
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--kernel', choices=['rbf', 'linear', 'poly'], default='poly')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--max_features', default='sqrt')
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--ufunc', choices=['tanh', 'sigmoid', 'relu'], default='lin')
    parser.add_argument('--pairwise_metric', choices=['euclidean', 'manhattan', 'cosine'], default='euclidean')


    args = parser.parse_args()

    cross_valid = False

    # ===============================
    # Apply dark subtraction to the data
    # ===============================
    path = os.path.join('Data', args.data)

    apply_dark_subtraction(path)

    # ===============================
    # Load Dataset
    # ===============================
    
    data_stores = ['CW14', 'Dar5']
    models = ['elm', 'logreg', 'rf', 'svm']

    print('\n Loading {} data...'.format(args.data))
    X, y, n_classes, num_input = load_dataset(args)

    # write all the data to a file
    with open("results.txt", "a") as f:
        f.write('{} {} \n'.format(args.model, args.data))
    
    # ===============================
    # Apply cross validation
    # ===============================

    print('\n Applying cross validation on {} model...'.format(args.model))

    y_pred, y_true = cross_val(args, X, y, n_classes, num_input)
        
    # ===============================
    # Perform Analysis
    # ===============================

    analysis(y_pred, y_true)

    