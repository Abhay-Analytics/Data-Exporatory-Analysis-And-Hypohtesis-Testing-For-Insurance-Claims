#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime


# In[2]:


claim_data = pd.read_csv('claims.csv')
customer_data = pd.read_csv('cust_demographics.csv')


# In[3]:


claim_data.head(2)


# In[4]:


claim_data.info()


# In[5]:


claim_data.dtypes


# In[6]:


customer_data.head(2)


# In[7]:


customer_data.info()


# In[8]:


customer_data.rename(columns={'CUST_ID':'customer_id'}, inplace=True)


# In[9]:


customer_data.head(2)


# In[10]:


merge_data = pd.merge(right=claim_data, left=customer_data, on='customer_id')


# In[11]:


merge_data.head(20)


# In[12]:


merge_data.info()


# In[13]:


merge_data['DateOfBirth'] = pd.to_datetime(merge_data['DateOfBirth'], format ='%d-%b-%y')
merge_data['claim_date'] = pd.to_datetime(merge_data['claim_date'], format ='%m/%d/%Y')

merge_data.head(20)

Observation: After converting the Date of Birth to Datetime format, it appears that some years are incorrect as they exceed the current year, indicating possible errors in the data entry.
# In[14]:


merge_data['DateOfBirth'] = merge_data.apply(lambda x: x['DateOfBirth'] - pd.DateOffset(days=365.25 * 100) if ((x['DateOfBirth'].year < 2070) and (x['DateOfBirth'].year > 2024)) else x['DateOfBirth'], axis=1)


# ## Q3
# 

# In[15]:


# Remove '-' characters from the 'Contact' column
merge_data['Contact'] = merge_data['Contact'].str.replace('-', '')


# Remove all non-numeric characters from the 'claim_amount' column
merge_data['claim_amount'] = merge_data['claim_amount'].str.replace('$', '') # if it is giving error, then remove the .str or vice vers

# Convert the 'claim_amount' column to numeric
merge_data['claim_amount'] = pd.to_numeric(merge_data['claim_amount'])


merge_data.head(20)


# # Q4

# In[16]:


# Create an alert flag for unreported injury claims directly
merge_data['unreported_injury'] = ((merge_data['claim_type'] == 'Injury only') & (merge_data['police_report'] == 'No')).astype(int)

# Display the first few rows to verify the changes
merge_data[['claim_type', 'police_report', 'unreported_injury']].head()


# # Q5

# In[17]:


# Sort the data by 'customer_id' and 'claim_date' in descending order
merge_data.sort_values(by=['customer_id', 'claim_date'], ascending=[True, False], inplace=True)

# Drop duplicates keeping the first (most recent) observation
merge_data.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)

# Reset the index
merge_data.reset_index(drop=True, inplace=True)

merge_data.head(5)


# # Q6

# In[18]:


missing_value = merge_data.isnull().sum()
print(missing_value)


# In[19]:


merge_data.head()


# In[20]:


# Impute missing values with mean for 'claim_amount' and mode for 'total_policy_claims'
merge_data['claim_amount'].fillna(merge_data['claim_amount'].mean(), inplace=True)
merge_data['total_policy_claims'].fillna(merge_data['total_policy_claims'].mode()[0], inplace=True)

# Impute missing values with mode for categorical variables and mean for continuous variables
for column in merge_data.columns:
    if merge_data[column].dtype == 'object':
        merge_data[column].fillna(merge_data[column].mode()[0], inplace=True)
    elif merge_data[column].dtype == 'float64':
        merge_data[column].fillna(merge_data[column].mean(), inplace=True)

# Recheck for missing values
missing_values_after_imputation = merge_data.isnull().sum()
print(missing_values_after_imputation)


# In[21]:


merge_data.head()


# # Q7

# In[22]:


merge_data['Age'] = (datetime.now() - merge_data['DateOfBirth']).dt.days // 365

def categorized_age(age):
    if age < 18:
        return 'Childern'
    if 18 <= age <= 30:
        return 'Youth'
    if 30 <= age <= 60:
        return 'Adult'
    else:
        return 'Senior'
    
merge_data['Age_category'] = merge_data['Age'].apply(categorized_age)

merge_data.head(50)


# # Q8

# In[23]:


# average amount claimed by customer for different segment
average_claim_amount = merge_data.groupby('Segment')['claim_amount'].mean()
print(average_claim_amount)


# In[24]:


merge_data.head()


# # Q9

# In[25]:


from datetime import timedelta

# dates 20 days prior to 1/10/2018
prior_date = datetime(2018,10,1) - timedelta(days=20)

# filtering the data
filter_data = merge_data[merge_data['claim_date'] <= prior_date]

# total claim amount
total_claim_amount = filter_data.groupby('incident_cause')['claim_amount'].sum()

print(total_claim_amount)


# # Q10

# In[26]:


location_data = merge_data[(merge_data['State'].isin(['TX', 'DE', 'AK'])) & (merge_data['Age_category'] == 'Adult')]

# Count the number of claims for driver-related issues and causes
driver_related_claims_count = location_data[location_data['incident_cause'].str.contains('Driver')].shape[0]

print('Number od adults from TX, DE and AK who claimed insurance for driver-related issues and causes:', driver_related_claims_count)


# # Q11

# In[27]:


# Aggregate claim amount based on gender and segment
agg_claim_amount = merge_data.groupby(['gender', 'Segment'])['claim_amount'].sum().reset_index()

# Create a pie chart
plt.figure(figsize=(10, 8))
sns.set_palette('pastel')
plt.title('Aggregated Claim Amount by Gender and Segment')
plt.pie(agg_claim_amount['claim_amount'], labels=agg_claim_amount.apply(lambda x: f"{x['gender']} - {x['Segment']}", axis=1), autopct='%1.1f%%', startangle=140)
plt.show()


# # Q12

# In[28]:


# Filter the data for driver-related issues
driver_related_data = merge_data[merge_data['incident_cause'].str.contains('Driver')]

# Group the data by gender and calculate the total claim amount
gender_claim_amount = driver_related_data.groupby('gender')['claim_amount'].sum().reset_index()

# Plot a bar chart to compare the total claim amount for driver-related issues between males and females
plt.figure(figsize=(10, 6))
sns.barplot(x='gender', y='claim_amount', data=gender_claim_amount)
plt.title('Total Claim Amount for Driver-Related Issues by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Claim Amount')
plt.show()


# # Q13

# In[29]:


# Filter the data for fraudulent policy claims
fraudulent_data = merge_data[merge_data['fraudulent'] == 'Yes']

# Group the data by age category and count the number of fraudulent claims
fraudulent_claims_by_age = fraudulent_data.groupby('Age_category')['claim_id'].count()

# Find the age group with the maximum fraudulent policy claims
max_fraudulent_age_group = fraudulent_claims_by_age.idxmax()

plt.figure(figsize=(10, 6))
sns.barplot(x=fraudulent_claims_by_age.values, y=fraudulent_claims_by_age.index, palette='viridis')
plt.title('Number of Fraudulent Policy Claims by Age Group')
plt.xlabel('Number of Fraudulent Claims')
plt.ylabel('Age Group')
plt.show()


# # Q14

# In[30]:


merge_data['claim_month'] = merge_data['claim_date'].dt.to_period('M')

# Group the data by month and year, calculate total claim amount
monthly_claim_amount = merge_data.groupby('claim_month')['claim_amount'].sum()

# Sort the data by month and year
monthly_claim_amount = monthly_claim_amount.sort_index()

# Plot the monthly trend of total claim amount
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_claim_amount.index.astype(str), y=monthly_claim_amount.values, marker='o')
plt.title('Monthly Trend of Total Claim Amount')
plt.xlabel('Month')
plt.ylabel('Total Claim Amount')
plt.xticks(rotation=45)
plt.show()


# # Q15

# In[31]:


# Group the data by gender, age category, and fraudulent status, calculate the average claim amount
avg_claim_amount = merge_data.groupby(['gender', 'Age_category', 'fraudulent'])['claim_amount'].mean().reset_index()

# Create a pivot table for plotting
pivot_table = avg_claim_amount.pivot_table(index=['gender', 'Age_category'], columns='fraudulent', values='claim_amount')

# Plot a facetted bar chart
pivot_table.plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title('Average Claim Amount by Gender and Age Category')
plt.xlabel('Gender, Age category')
plt.ylabel('Average Claim Amount')
plt.legend(title='Fraudulent', labels=['Non-Fraudulent', 'Fraudulent'])
plt.xticks(rotation=0)
plt.show()


# # Hypothesis Test
# 

# # Q16

# In[32]:


from scipy import stats


# To test whether there is any similarity in the amount claimed by males and females, we can use an independent samples t-test. This test is appropriate when comparing the means of two independent groups, in this case, males and females.
# 
# Null Hypothesis (H0): There is no significant difference in the amount claimed between males and females.
# 
# Alternative Hypothesis (HA): There is a significant difference in the amount claimed between males and females.
# 
# We choose the independent samples t-test because we are comparing the means of two groups (males and females) to determine if there is a statistically significant difference in the amount claimed. If the p-value is less than our chosen significance level (e.g., 0.05), we reject the null hypothesis and conclude that there is a significant difference in the amount claimed between males and females.

# In[33]:


# Filter the data for males and females
male_claims = merge_data.loc[merge_data['gender'] == 'Male', 'claim_amount']
female_claims = merge_data.loc[merge_data['gender'] == 'Female', 'claim_amount']

p_value = 0.05
    
# Perform independent samples t-test
t_stat, p_value = stats.ttest_ind(male_claims, female_claims)

print(f"P-value: {p_value:.4f}")

result = "There is a significant difference" if p_value < 0.05 else "There is no significant difference"
print(f"{result} in the amount claimed between males and females.")


# # Q17

# To test whether there is any relationship between Age_category and Segment, we can use a chi-squared test of independence. 
# This test is appropriate when we have categorical variables and want to determine if there is a relationship between them.
# 
# Null Hypothesis (H0): There is no relationship between Age category and segment.
# 
# Alternative Hypothesis (HA): There is a relationship between Age category and segment.
# 
# We choose the chi-squared test of independence because we are examining the relationship between two categorical variables, age category and segment. 
# If the p-value is less than our chosen significance level (e.g., 0.05), we reject the null hypothesis and conclude that there is a relationship between age category and segment.

# In[34]:


# create a contingency table to perform the chi-square test
contingency_table = pd.crosstab(merge_data['Age_category'], merge_data['Segment'])

# chi-square test of independence
chi_test = stats.chi2_contingency(observed = contingency_table)

p_value = 0.05

print(f'p_value : {p_value : .3f}')


result = "There is a relationship " if p_value < 0.05 else "There is no relationship"
print(f"{result} between Age_category and Segment.")



# # Q18
To test whether the current year has shown a significant rise in claim amounts compared to the 2016-17 
fiscal average of $10,000, we can perform a one-sample t-test. 
This test will compare the mean claim amount for the current year with the hypothesized mean of $10,000.

Null Hypothesis (H0): The mean claim amount for the current year is not significantly different from $10,000.
    
Alternative Hypothesis (HA): The mean claim amount for the current year is significantly different from $10,000.
# In[35]:


from scipy.stats import ttest_1samp

current_year_claims = merge_data[merge_data['claim_date'].dt.year == 2018]['claim_amount']

# Perform one-sample t-test
t_stat, p_value = ttest_1samp(current_year_claims, 10000)

# Print results
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant rise in claim amounts compared to the 2016-17 fiscal average.")
else:
    print("Fail to reject the null hypothesis. There is no significant rise in claim amounts compared to the 2016-17 fiscal average.")


# # Q19
To find the relationship between the categorical variables "region" and "Segment" using the merge_data table, we can use a chi-squared test of independence. This test will help us determine if there is a statistically significant association between the two variables.

Null Hypothesis (H0): There is no association between region and Segment.
Alternative Hypothesis (HA): There is an association between region and Segment.
# In[36]:


contengency_table = pd.crosstab(merge_data['State'], merge_data['Segment'])

# chi-square test of independence
chi_test = stats.chi2_contingency(observed = contingency_table)

p_value = 0.05

print(f'p_value : {p_value : .3f}')


result = "There is a association " if p_value < 0.05 else "There is no association"
print(f"{result} between State and Segment.")


# In[37]:


#Q19 Alternative


# In[38]:


from scipy.stats import f_oneway

age_groups = merge_data['Age_category'].unique()
insurance_claims_by_age_group = [merge_data[merge_data['Age_category'] == age_group]['total_policy_claims'] for age_group in age_groups]

# Perform ANOVA test
f_stat, p_value = f_oneway(*insurance_claims_by_age_group)

# Print results
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in insurance claims among different age groups.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in insurance claims among different age groups.")


# # Q20

# To test if there is any relationship between the total number of policy claims and the claimed amount, we can use a correlation analysis. Specifically, we can calculate the correlation coefficient between these two variables to determine if there is a linear relationship between them.
# 
# Null Hypothesis (H0): There is no linear relationship between the total number of policy claims and the claimed amount.
# 
# Alternative Hypothesis (HA): There is a linear relationship between the total number of policy claims and the claimed amount.

# In[39]:


total_policy_claims = merge_data['total_policy_claims']
claimed_amount = merge_data['claim_amount']

# Perform correlation analysis
correlation_coefficient, p_value = stats.pearsonr(total_policy_claims, claimed_amount)

# Print results
print(f"Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a linear relationship between the total number of policy claims and the claimed amount.")
else:
    print("Fail to reject the null hypothesis. There is no linear relationship between the total number of policy claims and the claimed amount.")


# In[ ]:




