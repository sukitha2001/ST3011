"""
Bank Subscription Prediction - Complete ML Pipeline with Hyperparameter Tuning
================================================================================
This script includes:
- Data preprocessing (encoding categorical variables)
- Train/test split (70/30)
- Feature scaling (StandardScaler)
- Base models training
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Final model comparison (Base vs Tuned)
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
    X, y, test_size=0.3, random_state=3011
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
# 7. BASE MODEL TRAINING (Without Tuning)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training Base Models (Without Tuning)")
print("=" * 60)

# Store base results
base_results = []

# Logistic Regression
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(random_state=3011, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Test Accuracy: {acc_lr:.4f}")
base_results.append({'Model': 'Logistic Regression', 'Accuracy': acc_lr})

# SVM
print("\n--- SVM ---")
svm_model = SVC(kernel='rbf', random_state=3011)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Test Accuracy: {acc_svm:.4f}")
base_results.append({'Model': 'SVM', 'Accuracy': acc_svm})

# Random Forest
print("\n--- Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=3011)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Accuracy: {acc_rf:.4f}")
base_results.append({'Model': 'Random Forest', 'Accuracy': acc_rf})

# XGBoost
print("\n--- XGBoost ---")
xgb_model = XGBClassifier(n_estimators=100, random_state=3011, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Test Accuracy: {acc_xgb:.4f}")
base_results.append({'Model': 'XGBoost', 'Accuracy': acc_xgb})

# AdaBoost
print("\n--- AdaBoost ---")
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=3011)
ada_model.fit(X_train_scaled, y_train)
y_pred_ada = ada_model.predict(X_test_scaled)
acc_ada = accuracy_score(y_test, y_pred_ada)
print(f"Test Accuracy: {acc_ada:.4f}")
base_results.append({'Model': 'AdaBoost', 'Accuracy': acc_ada})

# Gradient Boosting
print("\n--- Gradient Boosting ---")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=3011)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
acc_gb = accuracy_score(y_test, y_pred_gb)
print(f"Test Accuracy: {acc_gb:.4f}")
base_results.append({'Model': 'Gradient Boosting', 'Accuracy': acc_gb})

# ============================================================
# 8. HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Hyperparameter Tuning with GridSearchCV & RandomizedSearchCV")
print("=" * 60)

tuned_results = []

# ------------------------------------------------------------
# 8.1 Logistic Regression - GridSearchCV
# ------------------------------------------------------------
print("\n--- Tuning Logistic Regression (GridSearchCV) ---")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

lr_grid = GridSearchCV(
    LogisticRegression(random_state=3011, max_iter=1000),
    lr_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
lr_grid.fit(X_train_scaled, y_train)
print(f"Best Parameters: {lr_grid.best_params_}")
print(f"Best CV Score: {lr_grid.best_score_:.4f}")

y_pred_lr_tuned = lr_grid.predict(X_test_scaled)
acc_lr_tuned = accuracy_score(y_test, y_pred_lr_tuned)
print(f"Test Accuracy: {acc_lr_tuned:.4f}")
tuned_results.append({'Model': 'Logistic Regression (Tuned)', 'Accuracy': acc_lr_tuned, 'Best_Params': lr_grid.best_params_})

# ------------------------------------------------------------
# 8.2 SVM - GridSearchCV
# ------------------------------------------------------------
print("\n--- Tuning SVM (GridSearchCV) ---")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

svm_grid = GridSearchCV(
    SVC(random_state=3011),
    svm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)
print(f"Best Parameters: {svm_grid.best_params_}")
print(f"Best CV Score: {svm_grid.best_score_:.4f}")

y_pred_svm_tuned = svm_grid.predict(X_test_scaled)
acc_svm_tuned = accuracy_score(y_test, y_pred_svm_tuned)
print(f"Test Accuracy: {acc_svm_tuned:.4f}")
tuned_results.append({'Model': 'SVM (Tuned)', 'Accuracy': acc_svm_tuned, 'Best_Params': svm_grid.best_params_})

# ------------------------------------------------------------
# 8.3 Random Forest - RandomizedSearchCV
# ------------------------------------------------------------
print("\n--- Tuning Random Forest (RandomizedSearchCV) ---")
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=3011),
    rf_param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=3011,
    n_jobs=-1
)
rf_random.fit(X_train_scaled, y_train)
print(f"Best Parameters: {rf_random.best_params_}")
print(f"Best CV Score: {rf_random.best_score_:.4f}")

y_pred_rf_tuned = rf_random.predict(X_test_scaled)
acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
print(f"Test Accuracy: {acc_rf_tuned:.4f}")
tuned_results.append({'Model': 'Random Forest (Tuned)', 'Accuracy': acc_rf_tuned, 'Best_Params': rf_random.best_params_})

# ------------------------------------------------------------
# 8.4 XGBoost - RandomizedSearchCV
# ------------------------------------------------------------
print("\n--- Tuning XGBoost (RandomizedSearchCV) ---")
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=3011, eval_metric='logloss'),
    xgb_param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=3011,
    n_jobs=-1
)
xgb_random.fit(X_train_scaled, y_train)
print(f"Best Parameters: {xgb_random.best_params_}")
print(f"Best CV Score: {xgb_random.best_score_:.4f}")

y_pred_xgb_tuned = xgb_random.predict(X_test_scaled)
acc_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)
print(f"Test Accuracy: {acc_xgb_tuned:.4f}")
tuned_results.append({'Model': 'XGBoost (Tuned)', 'Accuracy': acc_xgb_tuned, 'Best_Params': xgb_random.best_params_})

# ------------------------------------------------------------
# 8.5 AdaBoost - GridSearchCV
# ------------------------------------------------------------
print("\n--- Tuning AdaBoost (GridSearchCV) ---")
ada_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}

ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=3011),
    ada_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
ada_grid.fit(X_train_scaled, y_train)
print(f"Best Parameters: {ada_grid.best_params_}")
print(f"Best CV Score: {ada_grid.best_score_:.4f}")

y_pred_ada_tuned = ada_grid.predict(X_test_scaled)
acc_ada_tuned = accuracy_score(y_test, y_pred_ada_tuned)
print(f"Test Accuracy: {acc_ada_tuned:.4f}")
tuned_results.append({'Model': 'AdaBoost (Tuned)', 'Accuracy': acc_ada_tuned, 'Best_Params': ada_grid.best_params_})

# ------------------------------------------------------------
# 8.6 Gradient Boosting - RandomizedSearchCV
# ------------------------------------------------------------
print("\n--- Tuning Gradient Boosting (RandomizedSearchCV) ---")
gb_param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=3011),
    gb_param_dist,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    random_state=3011,
    n_jobs=-1
)
gb_random.fit(X_train_scaled, y_train)
print(f"Best Parameters: {gb_random.best_params_}")
print(f"Best CV Score: {gb_random.best_score_:.4f}")

y_pred_gb_tuned = gb_random.predict(X_test_scaled)
acc_gb_tuned = accuracy_score(y_test, y_pred_gb_tuned)
print(f"Test Accuracy: {acc_gb_tuned:.4f}")
tuned_results.append({'Model': 'Gradient Boosting (Tuned)', 'Accuracy': acc_gb_tuned, 'Best_Params': gb_random.best_params_})

# ============================================================
# 9. MODEL COMPARISON - BASE VS TUNED
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Model Comparison - Base vs Tuned")
print("=" * 60)

# Base models comparison
print("\n--- Base Models ---")
base_df = pd.DataFrame(base_results).sort_values('Accuracy', ascending=False)
print(base_df.to_string(index=False))

# Tuned models comparison
print("\n--- Tuned Models ---")
tuned_df = pd.DataFrame(tuned_results)[['Model', 'Accuracy']].sort_values('Accuracy', ascending=False)
print(tuned_df.to_string(index=False))

# Combined comparison
print("\n--- Combined Comparison ---")
combined_results = []
for base, tuned in zip(base_results, tuned_results):
    combined_results.append({
        'Model': base['Model'],
        'Base Accuracy': base['Accuracy'],
        'Tuned Accuracy': tuned['Accuracy'],
        'Improvement': tuned['Accuracy'] - base['Accuracy']
    })

combined_df = pd.DataFrame(combined_results).sort_values('Tuned Accuracy', ascending=False)
print(combined_df.to_string(index=False))

# Best overall model
all_models = base_results + [{'Model': r['Model'], 'Accuracy': r['Accuracy']} for r in tuned_results]
all_df = pd.DataFrame(all_models).sort_values('Accuracy', ascending=False)
best_overall = all_df.iloc[0]
print(f"\n🏆 Best Overall Model: {best_overall['Model']} with {best_overall['Accuracy']:.4f} accuracy")

# ============================================================
# 10. DETAILED REPORT FOR BEST TUNED MODEL
# ============================================================
best_tuned = tuned_df.iloc[0]
print("\n" + "=" * 60)
print(f"STEP 9: Detailed Report - {best_tuned['Model']}")
print("=" * 60)

# Get predictions based on best tuned model
tuned_predictions = {
    'Logistic Regression (Tuned)': y_pred_lr_tuned,
    'SVM (Tuned)': y_pred_svm_tuned,
    'Random Forest (Tuned)': y_pred_rf_tuned,
    'XGBoost (Tuned)': y_pred_xgb_tuned,
    'AdaBoost (Tuned)': y_pred_ada_tuned,
    'Gradient Boosting (Tuned)': y_pred_gb_tuned
}

best_pred = tuned_predictions[best_tuned['Model']]

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# Print best parameters
for r in tuned_results:
    if r['Model'] == best_tuned['Model']:
        print(f"\nBest Hyperparameters:")
        for param, value in r['Best_Params'].items():
            print(f"  - {param}: {value}")

# ============================================================
# 11. VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 10: Generating Visualizations")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Base Models Comparison
ax1 = axes[0, 0]
colors_base = plt.cm.Blues(np.linspace(0.4, 0.8, len(base_df)))
bars1 = ax1.barh(base_df['Model'], base_df['Accuracy'], color=colors_base)
ax1.set_xlabel('Accuracy')
ax1.set_title('Base Models Comparison')
ax1.set_xlim(0.85, 0.95)
for bar, acc in zip(bars1, base_df['Accuracy']):
    ax1.text(acc + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontsize=10)

# Plot 2: Tuned Models Comparison
ax2 = axes[0, 1]
colors_tuned = plt.cm.Greens(np.linspace(0.4, 0.8, len(tuned_df)))
bars2 = ax2.barh(tuned_df['Model'], tuned_df['Accuracy'], color=colors_tuned)
ax2.set_xlabel('Accuracy')
ax2.set_title('Tuned Models Comparison')
ax2.set_xlim(0.85, 0.95)
for bar, acc in zip(bars2, tuned_df['Accuracy']):
    ax2.text(acc + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontsize=10)

# Plot 3: Base vs Tuned Comparison
ax3 = axes[1, 0]
x = np.arange(len(combined_df))
width = 0.35
bars3a = ax3.bar(x - width/2, combined_df['Base Accuracy'], width, label='Base', alpha=0.8)
bars3b = ax3.bar(x + width/2, combined_df['Tuned Accuracy'], width, label='Tuned', alpha=0.8)
ax3.set_xlabel('Models')
ax3.set_ylabel('Accuracy')
ax3.set_title('Base vs Tuned Performance')
ax3.set_xticks(x)
ax3.set_xticklabels(combined_df['Model'], rotation=45, ha='right')
ax3.legend()
ax3.set_ylim(0.85, 0.95)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Confusion Matrix Heatmap
ax4 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Not Subscribed', 'Subscribed'],
            yticklabels=['Not Subscribed', 'Subscribed'])
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')
ax4.set_title(f'Confusion Matrix - {best_tuned["Model"]}')

plt.tight_layout()
plt.savefig('model_comparison_tuned.png', dpi=150)
print("Visualization saved as 'model_comparison_tuned.png'")
plt.show()

print("\n" + "=" * 60)
print("Pipeline Complete!")
print("=" * 60)
