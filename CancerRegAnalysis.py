import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Load the dataset
data_path = 'cancer_reg.csv'
data = pd.read_csv(data_path, encoding='ISO-8859-1')

# Preprocessing steps
# Remove non-predictor columns as specified in the assignment
predictors = data.drop(columns=['avgAnnCount', 'avgDeathsPerYear', 'binnedInc', 'Geography', 'TARGET_deathRate'])

# Identify and remove columns with more than 10% missing data
missing_data = predictors.isnull().mean() * 100
columns_to_drop = missing_data[missing_data > 10].index.tolist()
predictors = predictors.drop(columns=columns_to_drop)

# Remove rows with any remaining missing data
predictors_clean = predictors.dropna()

# Calculate the mean and standard deviation for each column
means = predictors_clean.mean()
std_devs = predictors_clean.std()

# Standardize the predictor data
scaler = StandardScaler()
predictors_standardized = pd.DataFrame(scaler.fit_transform(predictors_clean), columns=predictors_clean.columns)

# Correlation analysis to select top predictors
response_correlations = data.corrwith(data['TARGET_deathRate']).drop('TARGET_deathRate')
strongest_positive = response_correlations.idxmax()
strongest_positive_value = response_correlations.max()
strongest_negative = response_correlations.idxmin()
strongest_negative_value = response_correlations.min()

# Prepare data for regression
X = predictors_standardized[response_correlations.abs().sort_values(ascending=False).head(16).index.tolist()]
X = sm.add_constant(X)  # adding a constant
y = data.loc[predictors_clean.index, 'TARGET_deathRate']  # response variable

# Model fitting
model = sm.OLS(y, X).fit()

# Output the results
print(model.summary())

# Display the strongest correlations
print(f"\nThe strongest positive correlation is between TARGET_deathRate and {strongest_positive} with a coefficient of {strongest_positive_value:.3f}.")
print(f"The strongest negative correlation is between TARGET_deathRate and {strongest_negative} with a coefficient of {strongest_negative_value:.3f}.")

# Residual plots (using matplotlib for visualization)
import matplotlib.pyplot as plt
import seaborn as sns

residuals = model.resid
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot for homoscedasticity
sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax[0])
ax[0].set_title('Residuals vs Fitted Values')
ax[0].set_xlabel('Fitted Values')
ax[0].set_ylabel('Residuals')
ax[0].axhline(0, color='red', linestyle='--')

# Histogram for residuals to check normality
sns.histplot(residuals, kde=True, ax=ax[1])
ax[1].set_title('Histogram of Residuals')
ax[1].set_xlabel('Residuals')

plt.tight_layout()
plt.show()
