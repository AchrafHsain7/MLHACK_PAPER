import numpy as np 
import sklearn as sk 
import json


# Load dataset
CIFAR_STANDARD = "/home/achraf/Research/MLHACK_PAPER/Datasets/cifar10-orig-hog-o6-c8-b1.npz"
data  = np.load(CIFAR_STANDARD)
X, Y = data["X"], data["Y"]
print(X.shape)

results = {}

# KNN hyperparameter Tune
print("===========================")
print("KNN Hyperparameter tune")
pipe = sk.pipeline.Pipeline([
("clf", sk.neighbors.KNeighborsClassifier(n_neighbors=5))
])
params = {
    'clf__n_neighbors': [3, 5, 9, 15]
}
inner_cv = sk.model_selection.StratifiedKFold(n_splits=5)
grid = sk.model_selection.GridSearchCV(pipe, params, cv=inner_cv, verbose=2)
grid.fit(X, Y)
print("SCORE:", grid.best_score_)
print("best params:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
results["KNN"] = [grid.best_params_, grid.best_score_]

# DT hyperparameter Tune
print("===========================")
print("DT Hyperparameter tune")
pipe = sk.pipeline.Pipeline([
    ("clf", sk.tree.DecisionTreeClassifier(max_depth=10))
])
params = {
    'clf__max_depth': [3, 5, 10, 15]
}
inner_cv = sk.model_selection.StratifiedKFold(n_splits=5)
grid = sk.model_selection.GridSearchCV(pipe, params, cv=inner_cv, verbose=2)
grid.fit(X, Y)
print("SCORE:", grid.best_score_)
print("best params:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
results["DT"] = [grid.best_params_, grid.best_score_]

# DT hyperparameter Tune
print("===========================")
print("LSVM Hyperparameter tune")
pipe = sk.pipeline.Pipeline([
        ("pca", sk.decomposition.PCA(n_components=0.95)),
        ("svm", sk.linear_model.SGDClassifier(loss="hinge"))
])
params = {
    'pca__n_components' : [0.7, 0.8, 0.9, 0.95]
}
inner_cv = sk.model_selection.StratifiedKFold(n_splits=5)
grid = sk.model_selection.GridSearchCV(pipe, params, cv=inner_cv, verbose=2)
grid.fit(X, Y)
print("SCORE:", grid.best_score_)
print("best params:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
results["LSVM"] = [grid.best_params_, grid.best_score_]


# DT hyperparameter Tune
print("===========================")
print("KSVM Hyperparameter tune")
pipe = sk.pipeline.Pipeline([
        ("pca", sk.decomposition.PCA(n_components=0.8)),
        ("svm", sk.svm.SVC(kernel="rbf", C=10, max_iter=500))
])

params = {
    'svm__C' : [0.1, 1, 10]
}
inner_cv = sk.model_selection.StratifiedKFold(n_splits=5)
grid = sk.model_selection.GridSearchCV(pipe, params, cv=inner_cv, verbose=2)
grid.fit(X, Y)
print("SCORE:", grid.best_score_)
print("best params:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
results["KSVM"] = [grid.best_params_, grid.best_score_]

with open("/home/achraf/Research/MLHACK_PAPER/JSONS/hyperparams.json", "w") as f:
    json.dump(results, f, indent=4) 
