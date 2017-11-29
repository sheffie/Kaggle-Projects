# This python script is an attempt to replicate the amazingly detailed "Exploring Survival on the Titanic" Kernel produced by Megan Risdal on Kaggle in Python. The original script uses R to analyze, visualize, impute and predict. The entire thought process is entirely Megan's, and this is merely an attempt at reproducing the concept she uses in a different programming language.

# Import necessary libraries
import pandas as pd  # Data Manipulation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import numpy as np
from patsy import dmatrices

sns.set(style="ticks")

# Import Data
df_train = pd.read_csv("aaa_train.csv")
df_test = pd.read_csv("aaa_test.csv")
df_full = pd.concat([df_train, df_test], axis=0)


# Inspect Data.
print(df_full.info())
# We can observe that we are working with 1309 observations of a total of 12 variables. While most of them are non-null, some have missing values: Age, Cabin, Embarked, and Fare. The survived column also has missing values, which is normal given that those are the values that we are trying to predict.
# The above hints to the fact that some Feature Engineering will be necessary in order to fill in the missing values.

print(df_train.info())
print(df_test.info())
# In the test set, the only missing values seem to be the Age (332/418), Fare (417/418) and Cabin (91/418).

print(df_full.describe())
# Using summary statistics, we can observe that ~38% of people contained in the dataset survived the disaster. The average fare is 33, although the currency is unknown. For the purpose of the exercise, we will assume that the fares are all provided in Pounds, i.e. in the same currency. The standard deviation on the fare is high. We will analyze normality of different variables at later stages during the exercise.

print(df_full.head())
# When browsing the data, there are two specific items that come to mind. One is the names of the passengers. These contain a hidden variable, the title. We can parse the name of the Passengers to determine their title, and add this as a given feature of our dataset. This might create a risk of co-correlation between independent variables, since there would inherently be a correlation between their gender and title, but we can explore this at a later stage. The second item is the Cabin number. Even though the data is spotty, we can assume that most of Class I passengers had cabins, while Class III didn't. Also, the cabins seem named according to the cabin number, and the specific deck where the cabin was situated. Wgile I do not believe the cabin number will be valuable to our model, the deck might give us a bit more information as to the proximity to the lifeboats. We will explore this in more detail at a later stage.


# Data Visualization. Let's visualize!

# First off, let's produce a scatterplot matrix of the data. Seaborn provides a handy function for this.
sns.pairplot(df_full, diag_kind="kde", diag_kws=dict(shade=True), palette="Greens_d")
plt.show()

# Visualizing the plot, there are three thoughts that come to mind. One is that we have a categorical variable represented in the PClass variable. This variable is presented as discrete and ordinal in the given Dataset, wo we will need to create dummy variables to ensure that we do not skew the model. Two is that the distribution of Fare is positively skewed, which will affect the model if we do not normalize the data. Three is that Passenger ID, if it was not made obvious by the inspection of the data above, will not yield any interesting information and is merely a list of numbers associated to the position of the Passenger within the Dataset.

# Let's drop Passenger ID from our Dataset.
df_full = df_full.drop('PassengerId', 1)

# It is well known that during the incident, women and children were prioritized to board the safety boats. Let's verify this.
# Scatter plot of age, sex and Survived (presented in hue)
sns.swarmplot(x="Age", y="Sex", hue="Survived", data=df_train)
plt.show()

# The swarmplot shows a clear correlation between age, sex, and whether or not passengers survived. From the scatterplot matrix that we had above, it also appears that we have a correlation between fare and survived. Let's visulize that.
sns.regplot(x="Survived", y="Fare", data=df_train)
plt.show()

# Although there does seem to be a correlation, the slope does not seem very high. At this point, we have identified three variables that may have an impact on whether or not the passenger survived: 1) Sex, 2) Age, 3) Fare. Can we find others? Let's do some Feature Engineering!

# Feature Engineering - Part 1, Titles.
# Fortunately, title extraction is made easy by the fact the they are standardized (i.e. contained in between a comma and a dot).
df_full["Title"] = df_full["Name"].str.replace("(.*, )|(\..*)", "")

# Now, let's count the number of Passengers for each title, per sex.
print(df_full.groupby(["Sex", "Title"]).agg({"Name": ["count"]}))

# Some of these titles are not interesting because they have few values. We can replace them.
replace_strings = {"Dona": "Other", "Lady": "Other", "the Countess": "Other", "Capt": "Other", "Don": "Other", "Jonkheer": "Other", "Major": "Other", "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Sir": "Mr", "Dr": "Other", "Rev": "Other", "Col": "Other"}
df_full["Title"] = df_full["Title"].replace(replace_strings)

# This leaves us with fewer titles to work with.
print(df_full.groupby(["Sex", "Title"]).agg({"Name": ["count"]}))
sns.swarmplot(x="Age", y="Title", hue="Survived", data=df_full)
plt.show()

# All right. It looks like again, age and survival are connected. Titles assigned to younger individuals ("Master" and "Miss") have a higher overall chance of survival. Onto the decks. Let's split the deck from the Cabin number.

df_full["Pdeck"] = df_full["Cabin"].str.replace("[0-9]", "").astype(str).str[0]
df_full["Pdeck"] = df_full["Pdeck"].str.replace("n", "no_data")
print(df_full.head())

sorted_df_full = df_full.sort_values(by=["Pdeck"])
sns.barplot(x="Pdeck", y="Survived", data=sorted_df_full, palette="Greens_d")
plt.show()

# The barplot does not show any clear correlation between Deck and Survival Rate. However, we can note that the passengers that are assigned a cabin number in the data have a higher chance of survival that those that do not.

# Feature Engineering - Part 1, Family Names.
# Let's extract distinct Surnames from our Dataset.
df_full["Surname"] = df_full["Name"].str.split('[,.]', expand=False).str[0]
print(df_full.head())

# Let's compute the family size variable
df_full["FamilySize"] = df_full["Parch"] + df_full["SibSp"] + 1

sns.countplot(x="FamilySize", hue="Survived", data=df_full, palette="Greens_d")
plt.show()

sns.barplot(x="FamilySize", y="Survived", data=df_full, palette="Greens_d")
plt.show()

# The data shows that singletons have a lower survival rate than small families(2-4). Large families (5-6) seem to have a lower survival rate as well.

# Missing Value Imputation
# Now that we've engineered three new features out of the data we already had, let's look at imputing missing values.
# Our of all the variables, the variables that are missing values are the following:
# - Cabin (295/1309)
# - Age (1046/1309)
# - Embarked (1307/1309)
# - Fare (1308/1309)
# Cabin has too many missing values for us to properly impute, so we won't be using this one.
# Embarked and Fare can probably be determined easily, since only one or two missing values are present for those.
# Age is missing about 20% of the data, so we will need to use imputation algorithms to compute missing values for age. (for examplem by using MICE)

# Embarked
# Embarked will probably be a factor of the ticket Fare and of the Passenger Class.
print(df_full[pd.isnull(df_full["Embarked"])])
# We can see that both of them payed 80 in Fare, and are in first class. Let's compare this to others.
f = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df_full)
plt.axhline(80, color='r', linestyle='--')
plt.show(f)

# Well. The missing Embarked information certainly coincides nicely with people embarking from Charbourg ('C'). We can replace the NA values for Embarked with 'C'
df_full["Embarked"] = df_full["Embarked"].fillna("C")
print(df_full.info())

# Fare
# Similarly, Fare might depend on where the passenger embarked, and what class they are travelling in.
print(df_full[pd.isnull(df_full["Fare"])])
# The passenger Embarked from Southampton (S) and is in third class.

df_s3 = df_full.ix[(df_full["Embarked"] == "S") & (df_full["Pclass"] == 3)]
f = sns.kdeplot(df_s3["Fare"], shade=True, color="b")
plt.axvline(df_s3["Fare"].dropna().median(), color='r', linestyle='--')
plt.show(f)
print(df_s3["Fare"].dropna().median())

# The median value for passengers embarking from Southampton in Third class is 8.05. We can use this as a proxy to replace the missing value for fare.
df_full["Fare"] = df_full["Fare"].fillna(df_s3["Fare"].dropna().median())
print(df_full.info())

# Age
# The best variable that comes to mind to get an approximation of age is Title. Master would imply younger individuals, so would Miss. The second would be Parch. In essence, the number of parents and childs may correlate with the age of the passenger. Let's verify.

sns.swarmplot(x="Parch", y="Age", hue="Survived", data=df_full)
plt.show()

# Yes, we can see a correlation between the Age and number of relatives. We can use this to impute our the Age variable for missing people.

df_full_impute = df_full.copy()
df_full_impute["Age"] = df_full_impute.groupby(["Title", "Parch"]).transform(lambda x: x.fillna(x.median()))
df_full_impute["Age"] = df_full_impute.groupby(["Title"]).transform(lambda x: x.fillna(x.median()))
print(df_full_impute.info())

# Let's look at the difference between the age distribution for both df_full and df_full_impute
plt.subplot(1, 2, 1)
plt.hist(df_full[df_full["Age"].notnull()]["Age"], facecolor="g")
plt.title("Original Data")

plt.subplot(1, 2, 2)
plt.hist(df_full_impute["Age"], facecolor="g")
plt.title("Imputed Data")

plt.show()

# Nothing seemed to have gone awfully awry, so let's go ahead and overwrite our original data.
df_full = df_full_impute

# Prediction
# Let's split our full data frame into the training and test set.
df_train = df_full[df_full.Survived.isin([1, 0])]
df_test = df_full[df_full.Survived.isnull()]
df_test["Survived"] = 0

y_train, X_train = dmatrices("Survived ~ Age + Embarked + Fare + Parch + Pclass + Sex + SibSp + Title + Sex:Age", df_train, return_type="dataframe")
y_test, X_test = dmatrices("Survived ~ Age + Embarked + Fare + Parch + Pclass + Sex + SibSp + Title + Sex:Age", df_test, return_type="dataframe")


parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
              'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
clf = RandomForestClassifier(**parameters)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_train.as_matrix(), y_train.values.ravel(), cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))
plt.show()

predict = clf.predict(X_test)
df_output = pd.DataFrame()
aux = pd.read_csv('aaa_test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = predict
df_output[['PassengerId', 'Survived']].to_csv('output.csv', index=False)
