# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:58:02 2025

@author: dopre2121
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import seaborn as sns
from sklearn.decomposition import PCA
from bankruptcy_final import X_c,y_c

#Downsampling
from imblearn.under_sampling import RandomUnderSampler
I = []
Accuracy = []
for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.3, random_state=42)
    
    rus = RandomUnderSampler(random_state=i)
    X_train_downsampled, y_train_downsampled = rus.fit_resample(X_train, y_train)
    
    #看平衡後0、1的數量
    #print(pd.Series(y_downsampled).value_counts())
   
    
    #Step 2: Apply PCA
    pca = PCA()
    pca.fit(X_train_downsampled)
    
    # Step 3: Get absolute loadings (contributions of each feature)
    loadings = np.abs(pca.components_)  # shape: (n_components, n_features)
    
    # Step 4: Score features by total contribution to top k components
    k_components = 5
    feature_scores = loadings[:k_components].sum(axis=0)
    
    # Step 5: Automatically drop features below threshold (e.g., mean contribution)
    threshold = np.mean(feature_scores)
    keep_mask = feature_scores >= threshold
    selected_indices = np.where(keep_mask)[0]
    
    # Use original feature names for clarity
    feature_names = X_train_downsampled.columns
    selected_features = feature_names[selected_indices]
    
    # Step 6: Reduce datasets to selected features
    X_train_selected = X_train_downsampled.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]
    
    df_1_train_down = pd.DataFrame(X_train_selected, columns=X_train_selected.columns)
    df_1_train_down['bankrupt'] = y_train_downsampled
    
    df_1_test_down = pd.DataFrame(X_test_selected, columns=X_test_selected.columns)
    df_1_test_down['bankrupt'] = y_test
    
    df_1_train_down.head()
    
    X_train = X_train_downsampled.values
    y_train = y_train_downsampled.values
    X_test = X_test.values
    y_test = y_test.values
    
    
    #random forest
    rf = RandomForestClassifier(random_state=10)

    # 設定要搜尋的參數網格
    parameters = {
        'n_estimators': [100, 200, 300],  # 樹的數量
        'max_depth': [3, 5, 7, 9],        # 樹的最大深度
        'min_samples_split': [2, 5, 10]   # 分裂節點所需的最小樣本數
    }

    # 使用GridSearchCV尋找最佳參數
    grid = GridSearchCV(rf, parameters, n_jobs=2, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    # 顯示最佳參數和分數
    print("\n最佳參數:", grid.best_params_)
    print("訓練集最佳分數:", grid.best_score_)

    # 使用最佳參數的模型
    best_rf = grid.best_estimator_

    Accuracy.append(grid.best_score_)
    I.append(i)
    
    
print(Accuracy)
print(f'mean_accuracy:{np.mean(Accuracy)}')
max_accuracy = np.max(Accuracy)
q = Accuracy.index(max_accuracy)
best_random_state = I[q]
print('best dwsampling')
print(f'random_state:{best_random_state}')



X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.3, random_state=42)
 
rus = RandomUnderSampler(random_state = best_random_state)
X_train_downsampled, y_train_downsampled = rus.fit_resample(X_train, y_train)
 
#看平衡後0、1的數量
print(pd.Series(y_train_downsampled).value_counts())

 
# Step 2: Apply PCA
pca = PCA()
pca.fit(X_train_downsampled)

# Step 3: Get absolute loadings (contributions of each feature)
loadings = np.abs(pca.components_)  # shape: (n_components, n_features)

# Step 4: Score features by total contribution to top k components
k_components = 5
feature_scores = loadings[:k_components].sum(axis=0)

# Step 5: Automatically drop features below threshold (e.g., mean contribution)
threshold = np.mean(feature_scores)
keep_mask = feature_scores >= threshold
selected_indices = np.where(keep_mask)[0]

# Use original feature names for clarity
feature_names = X_train_downsampled.columns
selected_features = feature_names[selected_indices]

# Step 6: Reduce datasets to selected features
X_train_selected = X_train_downsampled.iloc[:, selected_indices]
X_test_selected = X_test.iloc[:, selected_indices]

df_1_train_down = pd.DataFrame(X_train_downsampled, columns=X_train_selected.columns)
df_1_train_down['bankrupt'] = y_train_downsampled

df_1_test_down = pd.DataFrame(X_test_selected, columns=X_test_selected.columns)
df_1_test_down['bankrupt'] = y_test

df_1_train_down.head()
 
X_train = df_1_train_down.drop(columns=['bankrupt']).values
y_train = df_1_train_down['bankrupt'].values
X_test = df_1_test_down.drop(columns=['bankrupt']).values
y_test = df_1_test_down['bankrupt'].values


#random forest
rf = RandomForestClassifier(random_state=10)

# 設定要搜尋的參數網格
parameters = {
    'n_estimators': [100, 200, 300],  # 樹的數量
    'max_depth': [3, 5, 7, 9],        # 樹的最大深度
    'min_samples_split': [2, 5, 10]   # 分裂節點所需的最小樣本數
}

# 使用GridSearchCV尋找最佳參數
grid = GridSearchCV(rf, parameters, n_jobs=2, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# 顯示最佳參數和分數
print("最佳參數:", grid.best_params_)
print("訓練集最佳分數:", grid.best_score_)

# 使用最佳參數的模型
best_rf = grid.best_estimator_

y_pred_test = best_rf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_test)
print("測試集準確率:", test_accuracy)

cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm,annot = True,fmt = 'd',cmap = 'Blues'
            ,xticklabels = ['Predict0','Predict1']
            ,yticklabels = ['Actual0','Actual1'])
plt.title('Confusion Matrix_Random forest')
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()

importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': df_1_train_down.drop(columns=['bankrupt']).columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)
# print(feature_importance_df)

# 繪製特徵重要性圖
top_n = 15
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df.head(top_n),
    palette='viridis'
)
plt.title(f'Top {top_n} Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


