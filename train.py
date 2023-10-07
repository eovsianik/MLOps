import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from tqdm import tqdm

from mlops.model import NewModel

def model_train(
    model: nn.Module,
    X_train,
    y_train,
    optimizer: torch.optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    epochs=200,
):
    losses = []

    for i in (pbar := tqdm(range(epochs))):
        optimizer.zero_grad()

        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss)
        pbar.set_description(f"Epoch #{i + 1:2}, loss: {loss.item():10.8f}")

        loss.backward()
        optimizer.step()
    return model, losses


if __name__ == '__main__':
    X, y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    model = NewModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model, _ = model_train(model, X_train, y_train, optimizer, criterion)

    torch.save(model.state_dict(), 'modelstate.pt')