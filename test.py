from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('train.csv')
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Age'] = SimpleImputer().fit_transform(df['Age'].to_numpy().reshape(-1, 1))
df.dropna(axis='rows', inplace=True)

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=123)

print("Metrics for my decision tree classifier:")
dtc = DecisionTree()
dtc.fit(X_train, y_train)
preds_tree = dtc.predict(X_test)
pred_probs_tree = dtc.predict_proba(X_test)
print(f'accuracy = {accuracy_score(y_test, preds_tree)}')
print(f'roc_auc_score = {roc_auc_score(y_test, pred_probs_tree)}')

print()

print("Metrics for my random forest classifier:")
rfc = RandomForest(n_trees=70)
rfc.fit(X_train, y_train)
preds_forest = rfc.predict(X_test)
pred_probs_forest = rfc.predict_proba(X_test)
print(f'accuracy = {accuracy_score(y_test, preds_forest)}')
print(f'roc_auc_score = {roc_auc_score(y_test, pred_probs_forest)}')


