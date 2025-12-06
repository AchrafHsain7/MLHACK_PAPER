import numpy as np 
import sklearn as sk 
import json
from datetime import datetime


CIFAR10_ORIG = "/home/achraf/Research/MLHACK_PAPER/Datasets/cifar10-orig-hog.npz"
CIFAR10_FSGM = "/home/achraf/Research/MLHACK_PAPER/Datasets/cifar10-fsgm-hog.npz"
CIFAR10_PGD = "/home/achraf/Research/MLHACK_PAPER/Datasets/cifar10-pgd-hog.npz"



def train_attack(origX, origY, pipe, paths):
    results = {}
    pipe.fit(origX, origY)
    y_pred = pipe.predict(origX)
    accuracy = sk.metrics.accuracy_score(origY, y_pred)
    print("Original accuracy:", accuracy)
    results["orig_acc"] = accuracy

    cifar_fsgm = np.load(paths[1])
    cifar_fsgm_X, cifar_fsgm_Y = cifar_fsgm["X"], cifar_fsgm["Y"]
    y_pred = pipe.predict(cifar_fsgm_X)
    accuracy = sk.metrics.accuracy_score(cifar_fsgm_Y, y_pred)
    print("FSGM accuracy:", accuracy)
    results["fsgm_acc"] = accuracy

    cifar_pgd = np.load(paths[2])
    cifar_pgd_X, cifar_pgd_Y = cifar_pgd["X"], cifar_pgd["Y"]
    y_pred = pipe.predict(cifar_pgd_X)
    accuracy = sk.metrics.accuracy_score(cifar_pgd_Y, y_pred)
    print("PGD accuracy:", accuracy)
    results["pgd_acc"] = accuracy
    return results


def model_compare(paths):
    model_results = {}
    cifar_orig = np.load(paths[0])
    cifar_orig_X, cifar_orig_Y = cifar_orig["X"], cifar_orig["Y"]
    print("===========================")
    print("HOG + KNN Transferability Test")
    pipeKNN = sk.pipeline.Pipeline([
    ("clf", sk.neighbors.KNeighborsClassifier(n_neighbors=3))
    ])
    res = train_attack(cifar_orig_X, cifar_orig_Y, pipeKNN, paths)
    model_results["KNN"] = res

    print("===========================")
    print("HOG + Decision Tree Transferability Test")
    pipeDT = sk.pipeline.Pipeline([
    ("clf", sk.tree.DecisionTreeClassifier(max_depth=10))
    ])
    res = train_attack(cifar_orig_X, cifar_orig_Y, pipeDT, paths)
    model_results["DT"] = res

    # train SVM 
    print("===========================")
    print("HOG + Linear SVM Transferability Test")
    pipeLSVM = sk.pipeline.Pipeline([
        ("pca", sk.decomposition.PCA(n_components=0.9)),
        ("svm", sk.linear_model.SGDClassifier(loss="hinge"))
    ])
    res = train_attack(cifar_orig_X, cifar_orig_Y, pipeLSVM, paths)
    model_results["LSVM"] = res

    print("===========================")
    print("HOG + RBF SVM Transferability Test")
    pipeKSVM = sk.pipeline.Pipeline([
        ("pca", sk.decomposition.PCA(n_components=0.9)),
        ("svm", sk.svm.SVC(kernel="rbf", C=1, max_iter=1000))
    ])
    res = train_attack(cifar_orig_X, cifar_orig_Y, pipeKSVM, paths)
    model_results["KSVM"] = res
    
    return model_results


if __name__ == "__main__":
    log = {}
    with open("/home/achraf/Research/MLHACK_PAPER/JSONS/files.json", "r") as f:
        DATASETS_DIRS = json.load(f)
    for i, paths in enumerate(DATASETS_DIRS):
        paths = [p+".npz" for p in paths]
        run_data = model_compare(paths)
        run_data["DATE"] = str(datetime.now())
        run_data["DATASETS"] = paths
        log[f"EXPERIMENT{i}"] = run_data

    with open("/home/achraf/Research/MLHACK_PAPER/JSONS/experiment_results.json", "w") as f:
        json.dump(log, f, indent=4)



    
