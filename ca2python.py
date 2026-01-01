import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)

file_path = ("C:\\Users\\user\\Downloads\\world_happiness_report.csv")
df = pd.read_csv(file_path)

target_regression = 'Happiness Score'
df = df.dropna(subset=[target_regression])

df = df.drop(columns=[
    df.columns[0], 'Standard Error', 'Dystopia Residual'
], errors='ignore')

median_score = df[target_regression].median()
df['Happiness_Level'] = df[target_regression].apply(
    lambda x: 'High' if x >= median_score else 'Low'
)

target_classification = 'Happiness_Level'

numerical_features = [
    'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
    'Freedom', 'Trust (Government Corruption)', 'Generosity'
]

categorical_features = ['Region', 'year']

all_features = numerical_features + categorical_features

numerical_pipeline = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
]

categorical_pipeline = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(numerical_pipeline), numerical_features),
        ('cat', Pipeline(categorical_pipeline), categorical_features)
    ],
    remainder='drop'
)


print("--- Generating Visualizations ---")

corr_matrix = df[numerical_features + [target_regression]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    linewidths=0.5
)
plt.title('Correlation Heatmap of Happiness Factors and Score')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df[target_regression], kde=True, bins=20, color='teal')
plt.title('Distribution of World Happiness Scores')
plt.xlabel(target_regression)
plt.ylabel('Frequency (Country-Year Observations)')
plt.show()


plt.figure(figsize=(14, 7))
sns.boxplot(
    x='Region',
    y=target_regression,
    hue='Region',
    data=df,
    palette='Set3'
)
plt.xticks(rotation=45, ha='right')
plt.title('Happiness Score Distribution by World Region')
plt.xlabel('Region')
plt.ylabel('Happiness Score')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='Economy (GDP per Capita)',
    y=target_regression,
    data=df,
    hue=target_classification,
    palette='RdYlGn',
    s=70
)
plt.title('GDP per Capita vs. Happiness Score')
plt.xlabel('Economy (GDP per Capita)')
plt.ylabel('Happiness Score')
plt.legend(title='Happiness Level')
plt.show()


print("\n--- Running Linear Regression Model ---")

X_reg = df[all_features]
y_reg = df[target_regression]

Xtr_reg, Xte_reg, ytr_reg, yte_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
Xtr_p_reg = preprocessor.fit_transform(Xtr_reg)
Xte_p_reg = preprocessor.transform(Xte_reg)

regressor = LinearRegression()
regressor.fit(Xtr_p_reg, ytr_reg)
yp_reg = regressor.predict(Xte_p_reg)

R2_reg = r2_score(yte_reg, yp_reg)
RMSE_reg = np.sqrt(mean_squared_error(yte_reg, yp_reg))
MAE_reg = mean_absolute_error(yte_reg, yp_reg)


print("--- Running Decision Tree Classification Model ---")

Xc_cls = df[all_features]
yc_cls = df[target_classification]

Xtr_cls, Xte_cls, ytr_cls, yte_cls = train_test_split(
    Xc_cls, yc_cls, test_size=0.3, random_state=42
)
Xtr_p_cls = preprocessor.fit_transform(Xtr_cls)
Xte_p_cls = preprocessor.transform(Xte_cls)

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(Xtr_p_cls, ytr_cls)
yp_cls = classifier.predict(Xte_p_cls)

ACC_cls = accuracy_score(yte_cls, yp_cls)
F1_cls = f1_score(yte_cls, yp_cls, average="weighted", zero_division=0)

conf_matrix = confusion_matrix(yte_cls, yp_cls)
class_labels = np.unique(yc_cls)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.title('Confusion Matrix – Decision Tree Classifier')
plt.xlabel('Predicted Happiness Level')
plt.ylabel('Actual Happiness Level')
plt.show()


print("--- Running K-Means Clustering ---")

X_factors = df[numerical_features]

numeric_imputer_scaler = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

X_factors_cleaned_scaled = numeric_imputer_scaler.fit_transform(X_factors)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Factor_Cluster'] = kmeans.fit_predict(X_factors_cleaned_scaled)

cluster_analysis = (
    df.groupby('Factor_Cluster')[target_regression]
    .mean()
    .reset_index()
    .sort_values(by=target_regression, ascending=False)
)
cluster_analysis.columns = ['Cluster_ID', 'Average_Happiness_Score']

plt.figure(figsize=(8, 6))
sns.barplot(
    x='Cluster_ID',
    y='Average_Happiness_Score',
    data=cluster_analysis,
    palette='plasma',
    hue='Cluster_ID',
    legend=False
)
plt.title('Average Happiness Score by Factor Cluster')
plt.xlabel('Factor Cluster ID (K=3)')
plt.ylabel('Average Happiness Score')
plt.show()


print("\n=========================================================")
print("           WORLD HAPPINESS ANALYSIS RESULTS")
print("=========================================================")

print("\n--- Regression Results (Predicting Happiness Score) ---")
print(f"R-squared (R²)  : {R2_reg:.6f}")
print(f"RMSE            : {RMSE_reg:.6f}")
print(f"MAE             : {MAE_reg:.6f}")

print("\n--- Classification Results (Predicting High/Low Happiness) ---")
print(f"Accuracy        : {ACC_cls:.4f}")
print(f"F1 Score        : {F1_cls:.4f}")

print("\n--- Happiness Factor Cluster Analysis (K=3) ---")
print("Clusters grouped by average scores in Economy, Family, Health, Freedom, Trust, Generosity.")
print(cluster_analysis.to_string(index=False))
print("=========================================================")
