import os   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# from ydata_profiling import ProfileReport
from ydata.profiling import ProfileReport

os.environ['YDATA_LICENSE_KEY'] = 'd0d86879-1ac0-43b5-b604-f5e93848d95d'

df = pd.read_csv('data/Datasetprojpowerbi.csv')

profile = ProfileReport(df, title="University Student Complaint Report", explorative=True, outlier=True)

profile.to_file("university_student_complaint_report.html")


# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Display the percentage of missing values for each column
missing_percentage = df.isnull().mean() * 100
print(f"missing_percentage: {missing_percentage}")

# Option 1: Drop rows with missing values
df_cleaned = df.dropna()

# Option 2: Fill missing values (depending on the column type)
# Example: Fill missing GPA values with the mean GPA
df['Gpa'] = df['Gpa'].fillna(df['Gpa'].mean())

# Recheck missing values
df_cleaned.isnull().sum()


# Convert categorical variables to appropriate data types
df['Genre'] = df['Genre'].astype('category')
df['Gender'] = df['Gender'].astype('category')
df['Nationality'] = df['Nationality'].astype('category')

# Verify changes
df.dtypes

# Get descriptive statistics for numerical columns
df.describe()

# Check distribution of categorical columns
df['Genre'].value_counts()
df['Gender'].value_counts()
df['Nationality'].value_counts()


# Plot the distribution of complaint genres
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Genre', hue='Genre', palette='Set2', legend=False)
plt.title('Distribution of Complaint Genres')
plt.xticks(rotation=90)
plt.savefig("custom_report_image/distribution_of_complaint_genres.png")


# Plot the number of complaints by gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender', hue='Genre', palette='Set1')
plt.title('Complaints by Gender')
plt.savefig("custom_report_image/complaints_by_gender.png")

# Plot the number of complaints by nationality
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Nationality', hue='Nationality', palette='Set3', legend=False)
plt.title('Complaints by Nationality')
plt.xticks(rotation=90)
plt.savefig("custom_report_image/complaints_by_nationality.png")



# Plot the distribution of complaints by age
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Genre', multiple='stack', palette='Set2')
plt.title('Distribution of Complaints by Age')
plt.savefig("custom_report_image/distribution_of_complaints_by_age.png")

df["Count"] = pd.to_numeric(df['Count'], errors='coerce')

df_cleaned = df.dropna(subset=["Age", "Gpa", "Reports", "Count"])

df_cleaned[['Age', 'Gpa', 'Count']] = df_cleaned[['Age', 'Gpa', 'Count']].apply(pd.to_numeric)


# Correlation matrix for numerical columns
correlation_matrix = df_cleaned[['Age', 'Gpa', 'Count']].corr()

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig("custom_report_image/correlation_matrix_of_numerical_features.png")


# Plot the number of complaints over the years
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', hue='Year', palette='Set1', legend=True)
plt.title('Number of Complaints Over the Academic Years')
plt.savefig("custom_report_image/number_of_complaints_over_the_academic_years.png")


# Boxplot to analyze GPA distribution by complaint genre
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Genre', y='Gpa', palette='Set2')
plt.title('GPA Distribution by Complaint Genre')
plt.xticks(rotation=90)
plt.savefig("custom_report_image/gpa_distribution_by_complaint_genre.png")



# Countplot for complaints based on gender
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender', hue='Genre', palette='Set1')
plt.title('Complaints by Gender Across Genres')
plt.savefig("custom_report_image/complaints_by_gender_across_genres.png")



# Contingency table for the chi-square test
complaints_by_gender = pd.crosstab(df['Gender'], df['Genre'])

# Perform chi-square test
chi2_stat, p_val, dof, expected = stats.chi2_contingency(complaints_by_gender)

print(f"Chi-square Statistic: {chi2_stat}")
print(f"P-value: {p_val}")


with open("university_student_complaint_report.html", 'r') as file:
    html_file = file.read()


custom_plot_html = """
<h1 style="text-align: center;">Custom Plot Visualization</h1>
<div style="display: flex; justify-content: center;">
<div style="display: grid; gap: 10px;">
<img src="custom_report_image/distribution_of_complaint_genres.png" style="width: 100%; height:100%;" />
<img src="custom_report_image/complaints_by_gender_across_genres.png" style="width: 100%; height: 100%;" />
<img src="custom_report_image/complaints_by_nationality.png" style="width: 100%; height: 100%;" />
<img src="custom_report_image/correlation_matrix_of_numerical_features.png" style="width: 100%; height: 100%;" />
<img src="custom_report_image/distribution_of_complaint_genres.png" style="width: 100%; height: 100%;" />
<img src="custom_report_image/gpa_distribution_by_complaint_genre.png" style="width: 100%; height: 100%;" />
<img src="custom_report_image/number_of_complaints_over_the_academic_years.png" style="width: 100%; height: 100%;" />
</div>
</div>
"""

html_content = html_file.replace("</body>", custom_plot_html + "</body>")

with open("index.html", 'w') as file:
    file.write(html_content)


