# Importing Packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing



# Importing dataset
print("Dataset is")
dataset= pd.read_csv('SampleDataset.csv')
print(dataset)
print("Shape of Dataset is {}" .format(dataset.shape))
print(type(dataset))



# Accesing Independent columns
print("\n\n\nIndependent columns/features are ")
print("Accessing columns by index")
independent_cols = dataset.iloc[:,:-1]
print(independent_cols)
print(type(independent_cols))

print("\nAccessing columns by label names")
independent_cols = dataset.loc[:,["Age", "Country"]]
print(independent_cols)



# Treating missing values

#Finding number of missing values in each feature
print("\n\n\nNumber of null values in each field")
print(dataset.isnull().sum())

# Method-1 : Deleting rows having empty cells
deleted_dataset = dataset.dropna()
print("\nAfter deleting rows having empty cells")
print(deleted_dataset)
print(deleted_dataset.shape)

# Deleting rows having empty cells in particular columns
partly_deleted_dataset = dataset.dropna(subset = ["Age", "Salary"])
print("\nAfter deleting rows having empty cells in 'Age' and 'Salary' columns")
print(partly_deleted_dataset)
print(partly_deleted_dataset.shape)


# Method-2 : Imputing missing values
# Take mean of all the values in the column if it is of numeric type
# Take mode of all the values in the column if it is of non-numeric type

dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
dataset['Salary'].fillna(dataset['Salary'].median(), inplace=True)

dataset['Country'].fillna(dataset['Country'].mode()[0], inplace=True)

print(dataset)
print(dataset.shape)



# One-Hot Encoding
label_encoder= LabelEncoder()
print("\n\n\nAfter applying One Hot encoding")
dataset.iloc[:, 0] = label_encoder.fit_transform(dataset.iloc[:, 0])
dataset.iloc[:,-1] = label_encoder.fit_transform(dataset.iloc[:, -1])
print(dataset)



# Feature Scaling
# Method-1 : Standardize the data attributes
    # shifting the distribution of each attribute to have a mean of zero and a standard deviation of one
print("\n\n\nAfter performing Standardization")
standardized_dataset = preprocessing.scale(dataset)
print(standardized_dataset)

# Method-2 : Normalize the data attributes (range of 0 to 1)
print("\nAfter performing Normalization")
normlaized_dataset = preprocessing.normalize(dataset)
print(normlaized_dataset)
