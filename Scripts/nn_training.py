import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import sklearn as sk 
import json
from datetime import datetime


# This function was generated with the help of LLM's for speeding up experimentation :)
def nn_train(
    X_train,
    Y_train,
    n_iters=10,          # now n_iters = epochs
    lr=1e-2,
    batch_size=64,
    device="cuda"
):
    device = device if torch.cuda.is_available() else "cpu"

    # --- prepare tensors ---
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(Y_train, dtype=torch.long)

    # flatten if needed (important for images)
    if X.ndim > 2:
        X = X.view(X.size(0), -1)

    print("==========================================")
    print(X[:10])
    print("==========================================")

    # move dataset to CPU (DataLoader handles GPU later)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- model ---
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- training ---
    model.train()
    for epoch in range(n_iters):
        epoch_loss = 0.0

        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * Xb.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"epoch {epoch+1}/{n_iters}  loss={avg_loss:.4f}")

    return model

    




def train_attack(origX, origY, paths):
    results = {}
    device = "cuda"
    model = nn_train(origX, origY, n_iters=30)
    with torch.no_grad():
        X = torch.tensor(origX, dtype=torch.float32).to(device)
        logits = model(X).cpu()
        y_pred = torch.argmax(logits, dim=1).numpy()


    accuracy = sk.metrics.accuracy_score(origY, y_pred)
    print("Original accuracy:", accuracy)
    results["orig_acc"] = accuracy

    cifar_fsgm = np.load(paths[1])
    cifar_fsgm_X, cifar_fsgm_Y = cifar_fsgm["X"], cifar_fsgm["Y"]
    with torch.no_grad():
        X = torch.tensor(cifar_fsgm_X, dtype=torch.float32).to(device)
        logits = model(X).cpu()
        y_pred = torch.argmax(logits, dim=1).numpy()
    accuracy = sk.metrics.accuracy_score(cifar_fsgm_Y, y_pred)
    print("FSGM accuracy:", accuracy)
    results["fsgm_acc"] = accuracy

    cifar_pgd = np.load(paths[2])
    cifar_pgd_X, cifar_pgd_Y = cifar_pgd["X"], cifar_pgd["Y"]
    with torch.no_grad():
        X = torch.tensor(cifar_pgd_X, dtype=torch.float32).to(device)
        logits = model(X).cpu()
        y_pred = torch.argmax(logits, dim=1).numpy()
    accuracy = sk.metrics.accuracy_score(cifar_pgd_Y, y_pred)
    print("PGD accuracy:", accuracy)
    results["pgd_acc"] = accuracy
    return results



def model_compare(paths):
    model_results = {}
    cifar_orig = np.load(paths[0])
    cifar_orig_X, cifar_orig_Y = cifar_orig["X"], cifar_orig["Y"]
    print("===========================")
    print("HOG + ANN Transferability Test")
    res = train_attack(cifar_orig_X, cifar_orig_Y, paths)
    model_results["ANN"] = res
    return model_results





if __name__ == "__main__":
    log = {}
    with open("/home/achraf/Research/MLHACK_PAPER/JSONS/files.json", "r") as f:
        DATASETS_DIRS = json.load(f)
    for i, paths in enumerate(DATASETS_DIRS):
        paths = [p+".npz" for p in paths]
        print(paths)
        run_data = model_compare(paths)
        run_data["DATE"] = str(datetime.now())
        run_data["DATASETS"] = paths
        log[f"EXPERIMENT{i}"] = run_data
        if i==1:
            break

    print(log)

    # with open("/home/achraf/Research/MLHACK_PAPER/JSONS/experiment_results_ANN.json", "w") as f:
    #     json.dump(log, f, indent=4)
