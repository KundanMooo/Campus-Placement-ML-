import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Placement_Data_Full_Class.csv')

# Drop the 'sl_no' column as it's not needed for analysis
df = df.drop(columns='sl_no', axis=1)

# Display the first 10 rows of the dataset
df.head(10)

# Get the shape (number of rows and columns) of the dataset
df.shape

# Get the number of unique values for each categorical column
df.select_dtypes(include='object').nunique()

# Check for missing values and display the count of missing values
df.isna().sum()

# Check for missing values and display the proportion of missing values
df.isna().mean()

# Display information about the dataset
df.info()

# Extract categorical columns
dfcat = df.select_dtypes(include='object')

# Get a list of categorical column names
cat_col = df.select_dtypes(include='object').columns

# Calculate the percentage of missing values for each column and visualize with a bar plot
missing_percentage = (df.isna().mean() * 100).sort_values(ascending=False)
sns.barplot(x=missing_percentage.index, y=missing_percentage.values)

# Define a function to plot count plots for categorical columns
def plot_countplot(col):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Count Plot of {col}')
    plt.tight_layout()
    plt.close()

# Create subplots for count plots of categorical columns
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
for i, col in enumerate(cat_col):
    row_index = i // 4
    col_index = i % 4
    plot_countplot(col)
    sns.countplot(data=df, x=col, ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title(f'Count Plot of {col}')
plt.tight_layout()
plt.show()

# Extract numerical columns
dfnum = df.drop(dfcat.columns, axis=1)

# Get a list of numerical column names
num_col = dfnum.columns

# Define a function to plot different distributions for numerical columns
def plot_numerical_distribution(col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plots = [('stripplot', sns.stripplot), ('violinplot', sns.violinplot), ('kdeplot', sns.kdeplot)]
    for i, (plot_title, plot_func) in enumerate(plots):
        plot_func(data=dfnum, x=col, ax=axes[i])
        axes[i].set_title(f'{plot_title} of {col}')
    plt.tight_layout()
    plt.show()

# Create subplots for different distributions of numerical columns
for col in dfnum.columns:
    plot_numerical_distribution(col)

# Define a function to plot pie charts for categorical columns
def plot_pie_charts(df, cat_col):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, col in enumerate(cat_col):
        r = i // 3
        c = i % 3
        value_counts = df[col].value_counts()
        axes[r, c].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[r, c].set_title(f'Pie Chart of {col}')
    plt.tight_layout()
    plt.show()

# Plot pie charts for categorical columns
plot_pie_charts(df, cat_col)

# Define a function to plot box plots for numerical columns
def plot_boxplots(df, num_col):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axs = axs.flatten()
    for i, col in enumerate(num_col):
        sns.boxplot(x=col, data=df, ax=axs[i])
    fig.tight_layout()
    plt.show()

# Plot box plots for numerical columns
plot_boxplots(df, num_col)

# Define a function to plot box plots for numerical columns based on 'status'
def plot_boxplots_by_status(df, num_col):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axs = axs.flatten()
    for i, col in enumerate(num_col):
        sns.boxplot(y=col, x='status', data=df, ax=axs[i])
    fig.tight_layout()
    plt.show()

# Plot box plots for numerical columns based on 'status'
plot_boxplots_by_status(df, num_col)

# Define a function to plot violin plots for numerical columns based on 'status'
def plot_violinplots_by_status(df, num_col):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axs = axs.flatten()
    for i, col in enumerate(num_col):
        sns.violinplot(y=col, x='status', data=df, ax=axs[i])
    fig.tight_layout()
    plt.show()

# Plot violin plots for numerical columns based on 'status'
plot_violinplots_by_status(df, num_col)

# Define a function to plot violin plots for numerical columns
def plot_violinplots(df, num_col):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axs = axs.flatten()
    for i, col in enumerate(num_col):
        sns.violinplot(x=col, data=dfnum, ax=axs[i])
    fig.tight_layout()
    plt.show()

# Plot violin plots for numerical columns
plot_violinplots(df, num_col)

# Check for columns with missing values and their percentage
empty_col = df.isna().mean() * 100
empty_col[empty_col > 0]

# Fill missing values in the 'salary' column with the median value
df['salary'] = df['salary'].fillna(df['salary'].median())

# Check again for columns with missing values and their percentage
empty_col = df.isna().mean() * 100
empty_col[empty_col > 0]

# Display unique values for each categorical column
for col in cat_col:
    print(f'{col}:{df[col].unique()}')

# Import LabelEncoder for label encoding
from sklearn.preprocessing import LabelEncoder

# Create a new DataFrame to store label encoded categorical columns
dfcat_encoded = pd.DataFrame()

# Perform label encoding for each categorical column
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    dfcat_encoded[col] = label_encoder.fit_transform(dfcat[col])
    print(f'{col}:{dfcat_encoded[col].unique()}')

# Display the label encoded DataFrame
dfcat_encoded

# One-Hot Encoding for categorical columns
DF_encoded = pd.get_dummies(dfcat, columns=cat_col, drop_first=True)

# Display unique values in the one-hot encoded DataFrame
for i in DF_encoded:
    print(f'{i}:{DF_encoded[i].unique()}')

# Display a heatmap of the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), fmt='.2g', annot=True)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x = df.drop('status', axis=1)
y = df['status']

xtrn, xtst, ytrn, ytst = train_test_split(x, y, train_size=0.2)

# Remove outliers from the data using Z-score
from scipy.stats import zscore

z_scores = zscore(df[num_col])
df_no_outliers = df[(z_scores < 3).all(axis=1)]

# Import DecisionTreeClassifier and GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Create a DecisionTreeClassifier with balanced class weights
dt = DecisionTreeClassifier(class_weight='balanced')

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [2, 5, 7, 11, 13],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [2, 3, 4],
}

# Perform grid search for hyperparameter tuning
dt_grid = GridSearchCV(dt, param_grid, cv=4)
dt_grid.fit(xtrn, ytrn)

# Display the best parameters found by grid search
dt_grid.best_params_

# Display the accuracy score of the best model
print("Accuracy Score:", round(dt_grid.best_score_ * 100, 2), "%")

# Make predictions using the best model
ypred = dt_grid.predict(xtst)

# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Calculate and plot the confusion matrix
cm = confusion_matrix(ytst, ypred)
sns.heatmap(data=cm, annot=True, fmt='d', cmap='cool')

# Import roc_curve and auc
from sklearn.metrics import roc_curve, auc

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ytst, ypred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier with balanced class weights
rf = RandomForestClassifier(class_weight='balanced')

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 15, 17],
    'max_depth': [2, 5, 7, 11, 13, None],
    'min_samples_split': [2, 3, 5, 7],
    'min_samples_leaf': [2, 3, 4],
    'max_features': [0.2, 0.3, 0.5, 0.7]
}

# Perform grid search for hyperparameter tuning
rf_grid = GridSearchCV(rf, param_grid, cv=4)
rf_grid.fit(xtrn, ytrn)

# Display the best parameters found by grid search
rf_grid.best_params_

# Display the accuracy score of the best model
print("Accuracy Score:", round(rf_grid.best_score_ * 100, 2), "%")

# Make predictions using the best model
ypred = rf_grid.predict(xtst)

# Calculate and plot the confusion matrix
cm = confusion_matrix(ytst, ypred)
sns.heatmap(data=cm, annot=True, fmt='d', cmap='cool')

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ytst, ypred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create a LogisticRegression model
lg = LogisticRegression()

# Train the model on the training data
lg.fit(xtrn, ytrn)

# Make predictions on the testing data
ypred = lg.predict(xtst)

# Calculate the accuracy of the model
accuracy = round(accuracy_score(ytst, ypred), 2) * 100
print("Accuracy:", accuracy)

# Calculate and plot the confusion matrix
cm = confusion_matrix(ytst, ypred)
sns.heatmap(data=cm, annot=True, fmt='d', cmap='cool')

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ytst, ypred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()