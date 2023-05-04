import numpy as np
from scipy.linalg import qr
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class SuccessiveProjectionsAlgorithm:

    def eval(self, Xcal, ycal, selected_features, X_validation=None, y_validation=None):
        n_samples = Xcal.shape[0]
        pred_target_vals = np.zeros((n_samples, 1))
        e = None
            
        if X_validation is not None and y_validation is not None:
            # Use a separate validation set for validation
            Xcal_ones = np.hstack([np.ones((n_samples, 1)), Xcal[:, selected_features]])
            b = np.linalg.lstsq(Xcal_ones, ycal, rcond=None)[0]
            
            Xval_ones = np.hstack([np.ones((X_validation.shape[0], 1)), X_validation[:, selected_features]])
            pred_target_vals = Xval_ones.dot(b)
            
            e = y_validation - pred_target_vals
        else:
            # Use cross-validation
            for i in range(n_samples):
                # Remove item i from the training set
                cal = np.delete(np.arange(n_samples), i)
                X = Xcal[cal, selected_features]
                y = ycal[cal]
                xtest = Xcal[i, selected_features]
                
                X_ones = np.hstack([np.ones((n_samples - 1, 1)), X])
                b = np.linalg.lstsq(X_ones, y, rcond=None)[0]
                
                pred_target_vals[i] = np.hstack([np.ones(1), xtest]).dot(b)
                
            e = ycal - pred_target_vals

        return pred_target_vals, e

    def qr_proj(self, X, k, M):
        projected_X = X.copy()

        # Scale column k to make it the "largest" column
        max_norms = np.max(X ** 2, axis=0)

        projected_X[:, k] *= np.sqrt(2 * max_norms[k] / np.sum(X[:, k] ** 2))

        # Replace NaN with 0
        projected_X[np.isnan(projected_X)] = 0

        # Matrix partition with pivoting
        Q, R, column_order = qr(projected_X, mode='economic', pivoting=True)

        # Return the first M columns in the order specified by the pivot
        result = column_order[:M]
        if result.ndim == 1:
            result = result.reshape((-1, 1))
        return result.T

    def spa(self, X_train, y_train, m_min=1, m_max=None, X_validation=None, y_validation=None, autoscaling=1):
        # Get the number of samples and features in the training set
        num_samples, num_features = X_train.shape

        # Calculate the maximum number of features to select if not specified
        if m_max is None:
            if X_validation is None:
                m_max = min(num_samples - 1, num_features)
            else:
                m_max = min(num_samples - 2, num_features)

        # Calculate the normalization factor for the features
        if autoscaling == 1:
            normalization_factor = np.std(X_train, ddof=1, axis=0)
        else:
            normalization_factor = np.ones(num_features)

        # Normalize the training set
        X_train_norm = np.empty((num_samples, num_features))
        for k in range(num_features):
            # ignore warning
            with np.errstate(divide='ignore', invalid='ignore'):
                X_train_norm[:, k] = (X_train[:, k] - np.mean(X_train[:, k])) / normalization_factor[k]

        # Select the most orthogonal features using QR decomposition
        SEL = np.zeros((m_max, num_features))
        for k in range(num_features):
            SEL[:, k] = self.qr_proj(X_train_norm, k, m_max)

        # Calculate the prediction error sum of squares (PRESS) for each number of features and each feature
        PRESS = float('inf') * np.ones((m_max + 1, num_features))
        e = None
        for k in range(num_features):
            for m in range(m_min, m_max + 1):
                selected_features = SEL[:m, k].astype(np.int)
                _, e = self.eval(X_train, y_train, selected_features, X_validation, y_validation)
                PRESS[m, k] = np.conj(e).T.dot(e)

        # Select the number of features and the feature indices that minimize PRESS
        PRESS_min = np.min(PRESS, axis=0)
        m_sel = np.argmin(PRESS, axis=0)
        k_sel = np.argmin(PRESS_min)
        selected_features_2 = SEL[:m_sel[k_sel], k_sel].astype(np.int)

        # Perform least squares regression with the selected features
        X_train_sel = np.hstack([np.ones((num_samples, 1)), X_train[:, selected_features_2]])
        b = np.linalg.lstsq(X_train_sel, y_train, rcond=None)[0]
        std_deviation = np.std(X_train_sel, ddof=1, axis=0)

        # Calculate the relevance of each selected feature
        relev = np.abs(b * std_deviation.T)
        relev = relev[1:]

        # Sort the features by decreasing relevance
        index_increasing_relev = np.argsort(relev, axis=0)
        index_decreasing_relev = index_increasing_relev[::-1].reshape(1, -1)[0]

        # Calculate PRESS for each subset of features sorted by decreasing relevance
        PRESS_scree = np.empty(len(selected_features_2))
        e = None
        for i in range(len(selected_features_2)):
            selected_features = selected_features_2[index_decreasing_relev[:i + 1]]
            _, e = self.eval(X_train, y_train, selected_features, X_validation, y_validation)
            PRESS_scree[i] = np.conj(e).T.dot(e)

        # Calculate the critical value for PRESS using the F-distribution
        PRESS_scree_min = np.min(PRESS_scree)
        alpha = 0.25
        dof = len(e)
        fcrit = stats.f.ppf(1 - alpha, dof, dof)
        PRESS_crit = PRESS_scree_min * fcrit

        # Find the smallest number of features such that PRESS is below the critical value
        i_crit = np.min(np.nonzero(PRESS_scree < PRESS_crit))
        i_crit = max(m_min, i_crit)

        # Select the features with the highest relevance up to the critical number of features
        selected_features = selected_features_2[index_decreasing_relev[:i_crit]]

        return selected_features
    
    def select_features(self, path="./data/data.csv"):
        data = pd.read_csv(path)

        X = data.values[:, 1:]
        y = data.values[:, 0]

        # Data normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_normalized = scaler.fit_transform(X)

        # Model training-test split
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.5, random_state=0)

        # Feature selection
        var_selected = self.spa(X_train, y_train, m_min=8, m_max=30, X_validation=X_test, y_validation=y_test)

        return var_selected
                        