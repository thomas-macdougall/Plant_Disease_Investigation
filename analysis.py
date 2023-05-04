import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report,
                                cohen_kappa_score, matthews_corrcoef)

def analysis(y_pred, y_true):
    # combine the results from each fold into a single list 
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report (precision, recall, F1-score)
    cr = classification_report(y_true, y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    # write all the data to a file
    with open("results.txt", "a") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(cr)
        f.write("\n\nCohen's Kappa: {:.2f}".format(kappa))
        f.write("\n\nMatthews Correlation Coefficient: {:.2f}".format(mcc))
        f.write("\n\n\n")

    