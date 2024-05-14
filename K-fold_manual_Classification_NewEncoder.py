# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, TargetEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib

##-------------------------------------------------------------------------------------------------------##
# Define font properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 10}

# Set the font properties for matplotlib
matplotlib.rc('font', **font)

##-------------------------------------------------------------------------------------------------------##
# Load data of the dataset from the xlsx file.
file_name = 'Dataset/DatasetOfPaper.xlsx' #DatasetOfPaper

#Column names of the dataset
columsName = [
    'CVD type', 'Precursor T (˚C)', 'Substrate type', \
    'Growth T (˚C)', 'Growth time (min)', \
    'Growth P (torr)', 'Monolayer']
##-------------------------------------------------------------------------------------------------------##
def read_xlsx_file_convert_categorical_data(file_name, columsName):
    excel_data = pd.read_excel(file_name, sheet_name='Sheet1')

    # Read the values of the file in the dataframe
    data = pd.DataFrame(excel_data, columns=columsName)

    # convert_categorical_data
    le_CVD = LabelEncoder()
    # le_CVD = OrdinalEncoder()

    le_CVD_converted = le_CVD.fit_transform(data[["CVD type"]])
    print(le_CVD.classes_)
    data["CVD type"] = le_CVD_converted
    # print(le_CVD.__class__)


    le_substrate = LabelEncoder()
    # le_substrate = OrdinalEncoder()
    le_substrate_converted = le_substrate.fit_transform(data[["Substrate type"]])
    data["Substrate type"] = le_substrate_converted

    print(le_substrate.classes_)

    le_label = LabelEncoder()
    le_label_converted = le_label.fit_transform(data[["Monolayer"]])
    data["Monolayer"] = le_label_converted

    print(le_label.classes_)

    data = np.array(data)
    data = data.astype(np.float)
    return data, le_label

##-------------------------------------------------------------------------------------------------------##
def plot_save_ConfMat_classRep(allConfMat,classifierName,className,all_fold_pre_lbl):
    print("##-------------------------------------------"+ classifierName + '_classifier' +"------------------------------------------------##")
    fig, ax = plot_confusion_matrix(conf_mat=allConfMat,
                                    colorbar=True,
                                    show_absolute= False,
                                    show_normed=True,
                                    class_names=className,
                                    figsize=(12, 6))

    # Define file name for saving the plot
    fname = 'K-fold_ConfMat/ConfusionMat_AllFold_' + classifierName
    # Save the plot
    fig.savefig(fname)

    # Calculate accuracy of all fold
    sumElement = np.sum(allConfMat)
    accuracy = np.sum(allConfMat.diagonal()) / sumElement
    print('Mean Accuracy: ', accuracy)
    #plt.show()

    # Generate and print classification report of all folds
    report = classification_report(all_fold_act_lbl, all_fold_pre_lbl, target_names=className)
    print(report)

##-------------------------------------------------------------------------------------------------------##
data, label_encoder = read_xlsx_file_convert_categorical_data(file_name, columsName)
# split data into input and output columns
X, Y = data[:, :-1], data[:, -1]

# performin min-max scaling each continuous feature column to
# the range [0, 1]
cs = MinMaxScaler()
X = cs.fit_transform(X)

# Define per-fold score containers
all_fold_act_lbl = list()
all_fold_pre_lbl_RF = list()
all_fold_pre_lbl_KNN = list()
all_fold_pre_lbl_SVM = list()

# get class names of dataset
n_classes = len(label_encoder.classes_)
className = label_encoder.classes_

# Initialize three empty matrices to store k-fold confusion matrices for classifiers.
allConfMat_RF = np.zeros((n_classes, n_classes))
allConfMat_KNN  = np.zeros((n_classes, n_classes))
allConfMat_SVM = np.zeros((n_classes, n_classes))

# Define a Stratified Shuffle Split cross-validator
# Parameters:
#   n_splits: Number of re-shuffling and splitting iterations.
#   test_size: Represents the proportion of the dataset to include in the test split.
#   random_state: Controls the randomness of the training and testing data selection.
CV = StratifiedShuffleSplit(n_splits=5, test_size=0.30, random_state=1)

#declaration of classifiers
df = RandomForestClassifier(n_estimators=500, max_depth=4, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm = svm.SVC(kernel='rbf',C=2 )

# Initialize the fold index
fold_no = 1
# Generate fold indices and split data into train and test sets
for i, (train_index, test_index) in enumerate(CV.split(X, Y)):
    X_train = X[train_index]
    y_train = Y[train_index]
    X_test = X[test_index]
    y_test = Y[test_index]

    # Train each classifier with the training data for the current fold
    df.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #save actual label of test data
    all_fold_act_lbl.extend(y_test)

    #Test each trained classifier with the testing data for the current fold
    #for RF classifiers
    y_preds = df.predict(X_test)
    #report = classification_report(y_test, y_preds, target_names=label_encoder.classes_)
    Con_fold = confusion_matrix(y_test, y_preds)
    allConfMat_RF += np.array(Con_fold)
    all_fold_pre_lbl_RF.extend(y_preds)

    # for KNN classifiers
    y_preds = knn.predict(X_test)
    #report = classification_report(y_test, y_preds, target_names=label_encoder.classes_)
    Con_fold = confusion_matrix(y_test, y_preds)
    allConfMat_KNN += np.array(Con_fold)
    all_fold_pre_lbl_KNN.extend(y_preds)

    # for SVM classifiers
    y_preds = svm.predict(X_test)
    #report = classification_report(y_test, y_preds, target_names=label_encoder.classes_)
    Con_fold = confusion_matrix(y_test, y_preds)
    allConfMat_SVM += np.array(Con_fold)
    all_fold_pre_lbl_SVM.extend(y_preds)
    fold_no += 1

##-------------------------------------------------------------------------------------------------------##
#plot and save the confusion matrix for each classifier, then print the classification report.
plot_save_ConfMat_classRep(allConfMat_RF,'rf',className,all_fold_pre_lbl_RF)
plot_save_ConfMat_classRep(allConfMat_KNN,'knn',className,all_fold_pre_lbl_KNN)
plot_save_ConfMat_classRep(allConfMat_SVM,'svm',className,all_fold_pre_lbl_SVM)
