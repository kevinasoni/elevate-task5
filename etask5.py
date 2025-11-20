import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. Data Preparation
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Decision Tree Classifier (Unconstrained)
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Test Accuracy:", dt_acc)

# 3. Overfitting & Tree Depth
depths = range(1, 15)
accs = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    accs.append(accuracy_score(y_test, pred))
plt.figure(figsize=(8,5))
plt.plot(depths, accs, marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Test Accuracy')
plt.title('Decision Tree Depth vs Accuracy')
plt.savefig('dt_depth_vs_acc.png')

# 4. Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Test Accuracy:", rf_acc)

# 5. Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, rf.feature_importances_)
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig('rf_feat_imp.png')

# 6. Cross-validation evaluation
cv_dt = cross_val_score(dt_clf, X, y, cv=5)
cv_rf = cross_val_score(rf, X, y, cv=5)
print(f"Decision Tree CV Mean Acc: {cv_dt.mean():.3f}")
print(f"Random Forest CV Mean Acc: {cv_rf.mean():.3f}")

# 7. Confusion Matrices for documentation
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d')
plt.title('Decision Tree Confusion Matrix')
plt.savefig('dt_cm.png')
plt.close()

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.savefig('rf_cm.png')
plt.close()

# 8. (Optional) Export Decision Tree visualization for local annotation
export_graphviz(
    dt_clf,
    out_file="tree.dot",
    feature_names=X.columns,
    class_names=['No Disease', 'Disease'],
    filled=True, rounded=True, special_characters=True
)
# To generate tree visualization: run `dot -Tpng tree.dot -o dtree_viz.png` in terminal if Graphviz is installed

print("Done! Review all .png files for reporting.")
