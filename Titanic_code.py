# This python script is an attempt to replicate the amazingly detailed "Exploring Survival on the Titanic" Kernel produced by Megan Risdal on Kaggle in Python. The original script uses R to analyze, visualize, impute and predict. The entire thought process is entirely Megan's, and this is merely an attempt at reproducing the concept she uses in a different programming language.

# Import necessary libraries
import pandas as pd  # Data Manipulation
import seaborn as sns
import matplotlib.pyplot as plt

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
g = sns.pairplot(df_full, diag_kind="kde", diag_kws=dict(shade=True))
plt.show(g)

# Visualizing the plot, there are three thoughts that come to mind. One is that we have a categorical variable represented in the PClass variable. This variable is presented as discrete and ordinal in the given Dataset, wo we will need to create dummy variables to ensure that we do not skew the model. Two is that the distribution of Fare is positively skewed, which will affect the model if we do not normalize the data. Three is that Passenger ID, if it was not made obvious by the inspection of the data above, will not yield any interesting information and is merely a list of numbers associated to the position of the Passenger within the Dataset.

# Let's drop Passenger ID from our Dataset.
df_full = df_full.drop('PassengerId', 1)
