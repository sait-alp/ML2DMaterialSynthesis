# import the necessary packages
import pandas as pd
import numpy as np
import time
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import matplotlib

##-------------------------------------------------------------------------------------------------------##
# Define font properties
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 8}

# Set the font properties for matplotlib
matplotlib.rc('font', **font)
##-------------------------------------------------------------------------------------------------------##
## Load the xlsx file
file_name = 'Dataset/DatasetOfPaper.xlsx'
columsName = [
		'CVD type', 'Precursor T (˚C)','Substrate type', \
		'Growth T (˚C)', 'Growth time (min)', \
		'Growth P (torr)', 'Monolayer']

##-------------------------------------------------------------------------------------------------------##
def read_convert_xlsx(file_name,columsName):
	excel_data = pd.read_excel(file_name, sheet_name='Sheet1')

	# Read the values of the file in the dataframe
	data = pd.DataFrame(excel_data, columns=columsName)

	# convert_categorical_data
	le_CVD = LabelEncoder()
	le_CVD_converted = le_CVD.fit_transform(data[["CVD type"]])
	print(le_CVD.classes_)
	data["CVD type"] = le_CVD_converted
	print(le_CVD.__class__)

	le_substrate = LabelEncoder()
	le_substrate_converted = le_substrate.fit_transform(data[["Substrate type"]])
	data["Substrate type"] = le_substrate_converted
	print(le_substrate.classes_)

	le_label = LabelEncoder()
	le_label_converted = le_label.fit_transform(data[["Monolayer"]])
	data["Monolayer"] = le_label_converted
	print(le_label.classes_)

	data = np.array(data)
	data = data.astype(np.float)
	return data,le_label
##-------------------------------------------------------------------------------------------------------##
data,label_encoder = read_convert_xlsx(file_name,columsName)
# split data into input and output columns
X, Y = data[:, :-1], data[:, -1]

# performin min-max scaling each continuous feature column to
# the range [0, 1]
#cs = StandardScaler()
cs = MinMaxScaler()
X = cs.fit_transform(X)

start_time = time.time()
# Initialize a Random Forest Classifier model with 100 trees and a maximum depth of 5 for each tree
forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# Train the Random Forest model using the features (X) and the target labels (Y)
forest.fit(X, Y)

# Initialize a SHAP (SHapley Additive exPlanations) TreeExplainer to explain the model's predictions
# SHAP is a technique used to explain the output of machine learning models by attributing the prediction to input features
explainer = shap.TreeExplainer(forest)
# Compute SHAP values for each feature in the dataset (X)
shap_values = explainer.shap_values(X)
# Compute SHAP interaction values for pairwise feature interactions in the dataset (X)
shap_interaction_values = explainer.shap_interaction_values(X)
print(shap.summary_plot(shap_values[1], X, feature_names=columsName[0:6]))
print(shap.summary_plot(shap_values, features=X, feature_names=columsName[0:6], plot_type='bar'))
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")