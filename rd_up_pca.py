# -*- coding: utf-8 -*-
"""
Created on Mon May 26 22:12:20 2025

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
from bankruptcy_final import X_train,y_train,X_test,y_test,df_1_train_up,df_1_test_up

# 建立隨機森林模型
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
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print(feature_importance_df)



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