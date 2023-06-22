#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#check the envirenment
print ("in this poster design i will use Methane emissions (kt of CO2 equivalent) data obtained from world bank")


# In[ ]:


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computing and handling arrays
import sklearn  # For various machine learning algorithms and tools
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced data visualization
import tensorflow as tf  # For deep learning tasks (if using TensorFlow)
# or
# from tensorflow import keras  # For deep learning tasks (if using Keras)
import scipy  # For scientific and technical computing


# In[ ]:


# Read the data file
df = pd.read_csv("Methane emissions (kt of CO2 equivalent).csv")


# In[ ]:


df


# In[ ]:


# Display the first few rows of the dataframe
print(df.head(6))


# In[ ]:


# Display the first few rows of the dataframe
print(df.tail(5))


# In[ ]:


df.describe()


# In[ ]:


# Drop the desired variables
df = df.drop(['Country Name', 'Country Code', 'Series Name', 'Series Code'], axis=1)
df


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


# Remove spaces from variable names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

# Display the updated dataframe
print(df)


# In[ ]:


df


# In[ ]:


df.dtypes


# In[ ]:


# List of columns to convert to numeric
columns_to_convert = ['1990[YR1990]', '1995[YR1995]', '2000[YR2000]', '2001[YR2001]', '2002[YR2002]', '2003[YR2003]',
                      '2004[YR2004]', '2005[YR2005]', '2006[YR2006]', '2009[YR2009]', '2010[YR2010]', '2011[YR2011]',
                      '2012[YR2012]', '2013[YR2013]', '2014[YR2014]', '2015[YR2015]', '2016[YR2016]', '2017[YR2017]',
                      '2018[YR2018]', '2019[YR2019]']

# Convert the object columns to numeric
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Display the updated dataframe
print(df)


# In[ ]:


df.dtypes


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Extract the data columns
data_columns = df.columns[1:]

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Normalize the data
df[data_columns] = scaler.fit_transform(df[data_columns])

# Print the normalized data
print(df.head())


# In[ ]:


df.describe()


# In[ ]:


# Transpose the data
transposed_df = df.transpose()

# Print the transposed data
print(transposed_df.head())


# In[ ]:


def process_data(data_file):
    """
    Process the data from a CSV file.

    Parameters:
    data_file (str): The path to the CSV file containing the data.

    Returns:
    pandas.DataFrame: The processed DataFrame with normalized data.

    This function reads the data from a CSV file, normalizes the numerical
    columns, and returns the processed DataFrame. The CSV file is expected
    to have the following columns representing the years:
    1990[YR1990], 1995[YR1995], 2000[YR2000], ..., 2019[YR2019].

    Example usage:
    >>> processed_data = process_data("Methane emissions (kt of CO2 equivalent).csv")
    """
    # Implementation goes here
    pass


# In[ ]:


# Find null values in the dataframe
null_values = df.isnull().sum()

# Display the null values count
print(null_values)


# In[ ]:


# Drop rows with null values
df.dropna(inplace=True)


# In[ ]:


# Find duplicate rows
duplicate_rows = df[df.duplicated()]

# Print the duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)


# In[ ]:


null_values = df.isnull().sum()
print(null_values)


# In[ ]:


# Find duplicate rows
duplicate_rows = df[df.duplicated()]
# Print the duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)


# In[ ]:


# Select row 1 for visualization
row = df.iloc[1]

# Extract the column names and corresponding values
years = row.index
values = row.values

# Plot the time series data
plt.figure(figsize=(18, 10))
plt.plot(years, values)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Time Series Data for Albania')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[ ]:


# Select rows for visualization
rows = [1, 3, 5, 7, 9]  # Example: Afghanistan, American Samoa, Angola, Argentina, Aruba

# Iterate over the selected rows
for row_num in rows:
    # Extract the row corresponding to the country
    row = df.iloc[row_num]
    
    # Extract the country name
    country_name = df.index[row_num]
    
    # Extract the column names and corresponding values
    years = row.index
    values = row.values

    # Plot the time series data
    plt.figure(figsize=(12, 6))
    plt.plot(years, values)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title(f'Time Series Data for Row {row_num} ({country_name})')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Select rows for visualization
rows = [1, 3, 5, 7, 9]  # Example: Afghanistan, American Samoa, Angola, Argentina, Aruba

# Create a color map for the lines
colors = plt.cm.rainbow(np.linspace(0, 1, len(rows)))

# Create the plot
plt.figure(figsize=(22, 16))

# Iterate over the selected rows
for i, row_num in enumerate(rows):
    # Extract the row corresponding to the country
    row = df.iloc[row_num]
    
    # Extract the country name
    country_name = df.index[row_num]
    
    # Extract the column names and corresponding values
    years = row.index
    values = row.values

    # Plot the time series data with a different color for each country
    plt.plot(years, values, color=colors[i], label=country_name)

# Set the plot labels and title
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Time Series Data for Selected Countries')
plt.xticks(rotation=45)
plt.grid(True)

# Add a legend to differentiate the countries
plt.legend()

# Display the plot
plt.show()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
Y= df.iloc[: , 1: :]
Y


# In[ ]:


# Remove spaces from variable names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

# Drop the specified columns from the DataFrame
df = df.drop(['2001[YR2001]', '2002[YR2002]', '2003[YR2003]',
              '2004[YR2004]', '2006[YR2006]', '2009[YR2009]', '2011[YR2011]',
              '2012[YR2012]', '2013[YR2013]'], axis=1)

# Print the summary statistics of the modified DataFrame
print(df.describe())


# In[ ]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Print the correlation matrix
print(correlation_matrix)


# In[ ]:


# Calculate the correlation matrix
corr = df.corr()

# Set the figure size
plt.figure(figsize=(15, 10))

# Fill any missing values with 0
corr = corr.fillna(0)

# Create a heatmap of the correlation matrix
sns.heatmap(corr, annot=True, fmt=".3f", cmap="YlGnBu")

# Set the title of the plot
plt.title("Correlation Matrix")

# Display the plot
plt.show()


# In[ ]:


# Select the specific variables to include in the scatter matrix
selected_variables = ['1990[YR1990]', '1995[YR1995]', '2000[YR2000]']

# Create a DataFrame with only the selected variables
subset_df = df[selected_variables]

# Create a scatter matrix plot
pd.plotting.scatter_matrix(subset_df, figsize=(18, 18), s=5, alpha=0.8)

# Set the plot title
plt.title("Scatter Matrix Plot")

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[ ]:


# Select the specific variables to include in the scatter matrix
selected_variables = ['1990[YR1990]', '2000[YR2000]', '2010[YR2010]']

# Create a DataFrame with only the selected variables
subset_df = df[selected_variables]

# Define a color map for the scatter matrix plot
colors = ['red', 'green', 'blue']

# Create a scatter matrix plot
scatter_matrix = pd.plotting.scatter_matrix(subset_df, figsize=(18, 18), s=5, alpha=0.8)

# Set the plot title
plt.suptitle("Scatter Matrix Plot for 1990, 2000, and 2010")

# Adjust the spacing between subplots
plt.tight_layout()

# Modify the color of each subplot
for i in range(len(selected_variables)):
    for j in range(len(selected_variables)):
        if i != j:
            ax = scatter_matrix[i, j]
            ax.spines['bottom'].set_color(colors[i])
            ax.spines['top'].set_color(colors[i])
            ax.spines['left'].set_color(colors[j])
            ax.spines['right'].set_color(colors[j])
        else:
            ax = scatter_matrix[i, j]
            ax.hist(subset_df.iloc[:, i], color=colors[i])

# Display the plot
plt.show()


# In[ ]:


# Create a scatter matrix plot
pd.plotting.scatter_matrix(df, figsize=(18, 18), s=5, alpha=0.8)

# Set the plot title
plt.title("Scatter Matrix Plot")

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[ ]:


from scipy.optimize import curve_fit

# Define the function to fit the curve
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate sample data
x = np.linspace(0, 10, 100)
y = func(x, 2.5, 1.3, 0.5) + np.random.normal(0, 0.2, size=x.shape)

# Fit the curve to the data
popt, pcov = curve_fit(func, x, y)

# Extract the optimized parameters
a_opt, b_opt, c_opt = popt

# Generate fitted curve using optimized parameters
y_fit = func(x, a_opt, b_opt, c_opt)

# Plot the original data and fitted curve
plt.figure(figsize=(15, 10))
plt.plot(x, y, 'bo', label='Data')
plt.plot(x, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# In[ ]:




