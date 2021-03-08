## Capstone-Project<br>
# Table of Contents<br><br>

1.Project Overview<br>
2.Project Components<br>
3.Installation<br>
4.File Descriptions<br>
5.Instructions<br>
6.Results<br>
7.Acknowledgements<br>

## Project Overview<br>
In this project, I had analyzed demographics data for customers of a mail-order sales company in Germany, compared it against demographics information for the general population. I had used unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, I had applied what I've learned on a third dataset with demographics information for targets of a marketing campaign for the company, and used a trained model to predict which individuals are most likely to convert into becoming customers for the company. The data used has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task. <br>

# Project Components
The project has three major parts: the customer segmentation report, the supervised learning model, and the Kaggle Competition.<br>

Customer Segmentation Report: This part used the unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company.<br>

Supervised Learning Model: This part would use the third dataset with attributes from targets of a mail order campaign and use the previous analysis to build a machine learning model that predicts whether or not each individual will respond to the campaign.<br><br>

Kaggle Competition: Once best performing model would be identified, it would be used to make predictions on the campaign data as part of a Kaggle Competition. We would rank the individuals by how likely they are to convert to being a customer, and see how our modeling skills measure up against the fellow students.<br>


## Installation <br>
a.The code should run with no issues using Python versions 3.*.<br>
b.No extra besides the built-in libraries from Anaconda needed to run this project  <br>
c.Data Processing & Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn <br>
d.Data Visualization: Matplotlib, Seaborn<br><br>

## File Descriptions<br>
There are four data files associated with this project:<br><br><br>

A.Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).<br><br>
B.Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).<br>
C.Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).<br>
D.Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).<br>

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company. The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition. Otherwise, all of the remaining columns are the same between the three data files.<br>

For more information about the columns depicted in the files above, one can refer to two additional meta-data files(.xlsx) provided and one manually created file(.csv). <br>

a.DIAS Information Levels - Attributes 2017.xlsx: is a top-level list of attributes and descriptions, organized by informational category. <br>
b.DIAS Attributes - Values 2017.xlsx: is a detailed mapping of data values for each feature in alphabetical order.<br>
c.feature_complete.csv: is a feature summary file and is created manually using the above 2 files mapping all the attributes with information level, type & missing_or_unknown information to assist in our analysis.<br>

## Instructions <br>

1.Since the data files are not available publicly as dataset are highly proprietary, the Jupyter notebook is just for exploration. An HTML file has been provided as another means to check the notebook.<br>

2.feature_complete.csv should be used for data pre-processing.<br>

3.The Jupyter notebook produced a file called kaggle_submission_file.csv which was submitted to Kaggle for the in-class competition.<br>

## Results
a.It was necessary to apply the majority of CRISP-DM (CRoss-Industry Standard Process for Data Mining) to both parts of the project. Business Understanding was slipped in as part of the problem description, but Data Understanding, Data Preparation, Modelling and Evaluation had to be developed from scratch. The general rule that Data Preparation is the most time consuming part in the process could be verified once again. <br>

b.Part 1 presented how unsupervised learning techniques — namely PCA and Clustering with KMeans — were applied to distinguish groups of individuals that best describe the core customer base of the mail-order company. A supervised learning model, in form of LogisticRegression, then helped to identify the main characteristics of these individuals.<br>

c.Part 2 presented the straightforward way of building a supervised learning model. The base performance of various classifiers was determined. With the help of GridSearchCV the most promising one — GradientBoostingClassifier — was fitted respectively tuned to the training dataset (considering stratified cross-validation) and its performance was evaluated via ROC AUC. A short analysis of the most important features completed the model creation before using it for predicting on the testing dataset which individuals of a marketing campaign are most likely to convert into becoming customers. <br>

d.Part 3 presented the Kaggle in-class competition and kaggle_submission_file.csv was submitted to obtain a kaggle score which is roc_auc score on the testing data. Through Kaggle score, position of the participant in the leaderboard was determined. <br>

## Notes on Class Imbalance <br>
a.Since there is a large output class imbalance, predicting individual classes and using accuracy does not seem to be an appropriate performance evaluation method. Instead, the model will be using ROC-AUC to evaluate performance. <br>

b.Aside from the Kaggle competition using ROC-AUC as the score, the metric is suitable for binary classification problems such as this. ROC curves give us the ability to assess the performance of the classifier over its entire operating range. The most widely-used measure is the area under the curve (AUC). The AUC can be used to compare the performance of two or more classifiers. A single threshold can be selected and the classifiers' performance at that point compared, or the overall performance can be compared by considering the AUC". Compared to the F1 score, the ROC does not require optimizing a threshold for each label. <br>

## Acknowledgements <br>
This project was completed as part of the Udacity Data Scientist Nanodegree. The data was originally sourced by Udacity from Arvato Financial Solutions, a Bertelsmann subsidiary. <br>

Github Link - https://github.com/Shivaae/Capstone-Project

