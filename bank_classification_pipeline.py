"""
Bank Subscription Prediction - Complete ML Pipeline
====================================================
This script includes:
- Data preprocessing (encoding categorical variables)
- Train/test split (70/30)
- Feature scaling (StandardScaler)
- Multiple classification models
- Model evaluation and comparison
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

bank_df = pd.read_csv("week5-Data/bank-sample.csv")
print(f"Dataset shape: {bank_df.shape}")
print(f"Columns: {list(bank_df.columns)}")
print("\nFirst 5 rows:")
print(bank_df.head())

# ============================================================
# 3. DATA PREPROCESSING - ENCODING CATEGORICAL VARIABLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Encoding Categorical Variables")
print("=" * 60)

# Create a copy
bank_df_encoded = bank_df.copy()

# Identify categorical columns
categorical_cols = bank_df.select_dtypes(include=['object']).columns
print(f"Categorical columns: {list(categorical_cols)}")

# Method 1: Label Encoding for binary categorical variables (yes/no)
binary_cols = ['default', 'housing', 'loan', 'subscribed']
le = LabelEncoder()

for col in binary_cols:
    bank_df_encoded[col] = le.fit_transform(bank_df_encoded[col])
print(f"\nLabel encoded columns: {binary_cols}")

# Method 2: One-Hot Encoding for multi-class categorical variables
multi_class_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
bank_df_encoded = pd.get_dummies(bank_df_encoded, columns=multi_class_cols, drop_first=True)
print(f"One-hot encoded columns: {multi_class_cols}")
print(f"Shape after encoding: {bank_df_encoded.shape}")

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Separating Features and Target")
print("=" * 60)

X = bank_df_encoded.drop('subscribed', axis=1)
y = bank_df_encoded['subscribed']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ============================================================
# 5. TRAIN/TEST SPLIT (70/30)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split (70/30)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ============================================================
# 6. FEATURE SCALING (StandardScaler on numerical columns only)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Feature Scaling (StandardScaler)")
print("=" * 60)

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
print(f"Numerical columns to scale: {numerical_cols}")

scaler = StandardScaler()

# Fit on train, transform both
X_train_scaled = X_train.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

X_test_scaled = X_test.copy()
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("Scaling complete!")

# ============================================================
# 7. MODEL TRAINING AND EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Models")
print("=" * 60)

# Store results
results = []

# ------------------------------------------------------------
# 7.1 Logistic Regression
# ------------------------------------------------------------
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Test Accuracy: {acc_lr:.4f}")
results.append({'Model': 'Logistic Regression', 'Accuracy': acc_lr})

# ------------------------------------------------------------
# 7.2 SVM
# ------------------------------------------------------------
print("\n--- SVM ---")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Test Accuracy: {acc_svm:.4f}")
results.append({'Model': 'SVM', 'Accuracy': acc_svm})

# ------------------------------------------------------------
# 7.3 Random Forest
# ------------------------------------------------------------
print("\n--- Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy: {acc_rf:.4f}")
results.append({'Model': 'Random Forest', 'Accuracy': acc_rf})

# ------------------------------------------------------------
# 7.4 XGBoost
# ------------------------------------------------------------
print("\n--- XGBoost ---")
xgb_model = XGBClassifier(n_estimators=1000, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Test Accuracy: {acc_xgb:.4f}")
results.append({'Model': 'XGBoost', 'Accuracy': acc_xgb})

# ------------------------------------------------------------
# 7.5 AdaBoost
# ------------------------------------------------------------
print("\n--- AdaBoost ---")
ada_model = AdaBoostClassifier(n_estimators=1000, learning_rate=1.0, random_state=42)
ada_model.fit(X_train_scaled, y_train)
y_pred_ada = ada_model.predict(X_test_scaled)
acc_ada = accuracy_score(y_test, y_pred_ada)
print(f"Test Accuracy: {acc_ada:.4f}")
results.append({'Model': 'AdaBoost', 'Accuracy': acc_ada})

# ------------------------------------------------------------
# 7.6 Gradient Boosting
# ------------------------------------------------------------
print("\n--- Gradient Boosting ---")
gb_model = GradientBoostingClassifier(n_estimators=1000, random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"Test Accuracy: {acc_gb:.4f}")
results.append({'Model': 'Gradient Boosting', 'Accuracy': acc_gb})

# ============================================================
# 8. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Model Comparison")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n" + results_df.to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']} with {best_model['Accuracy']:.4f} accuracy")

# ============================================================
# 9. DETAILED REPORT FOR BEST MODEL
# ============================================================
print("\n" + "=" * 60)
print(f"STEP 8: Detailed Report - {best_model['Model']}")
print("=" * 60)

# Get predictions based on best model
best_predictions = {
    'Logistic Regression': y_pred_lr,
    'SVM': y_pred_svm,
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb,
    'AdaBoost': y_pred_ada,
    'Gradient Boosting': y_pred_gb
}

best_pred = best_predictions[best_model['Model']]

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# ============================================================
# 10. VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Generating Visualizations")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Comparison Bar Chart
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_df)))
bars = ax1.barh(results_df['Model'], results_df['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy')
ax1.set_title('Model Comparison')
ax1.set_xlim(0.8, 1.0)
for bar, acc in zip(bars, results_df['Accuracy']):
    ax1.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontsize=10)

# Plot 2: Confusion Matrix Heatmap
ax2 = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Not Subscribed', 'Subscribed'],
            yticklabels=['Not Subscribed', 'Subscribed'])
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')
ax2.set_title(f'Confusion Matrix - {best_model["Model"]}')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
print("Visualization saved as 'model_comparison.png'")
plt.show()

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
