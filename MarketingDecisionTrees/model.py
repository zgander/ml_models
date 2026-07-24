import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("./data/cleaned_marketing.csv")

df = pd.DataFrame(data)

x = df.drop('y',axis=1)
y = df['y']

x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    "max_depth": [3,5,7,10,15],
    "min_samples_split": [2,5,10,20,50],
    "min_samples_leaf": [1,2,5,10,20]
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=10, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Recall:", grid_search.best_score_)

best_tree = grid_search.best_estimator_

predictions_train = best_tree.predict(x_train)
predictions_test = best_tree.predict(x_test)

tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, predictions_train).ravel().tolist()
tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, predictions_test).ravel().tolist()

print("\nTraining Confusion Matrix (TN, FP, FN, TP): ",tn_train,fp_train,fn_train,tp_train)
print("Training Accuracy Score: ",accuracy_score(y_train,predictions_train))
print("Training Precision: ",tp_train/(tp_train + fp_train))
print("Training Recall: ",tp_train/(tp_train + fn_train))
print("F1 Score: ",2 * (tp_train/(tp_train + fp_train)) * (tp_train/(tp_train + fn_train)) / ((tp_train/(tp_train + fp_train)) + (tp_train/(tp_train + fn_train))))

print("\nTest Confusion Matrix (TN, FP, FN, TP): ",tn_test,fp_test,fn_test,tp_test)
print("Test Accuracy Score: ",accuracy_score(y_test,predictions_test))
print("Test Precision: ",tp_test/(tp_test + fp_test))
print("Test Recall: ",tp_test/(tp_test + fn_test))
print("F1 Score: ",2 * (tp_test/(tp_test + fp_test)) * (tp_test/(tp_test + fn_test)) / ((tp_test/(tp_test + fp_test)) + (tp_test/(tp_test + fn_test))))