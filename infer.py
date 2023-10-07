import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from mlops.model import NewModel

def model_predict(model: nn.Module, X_test):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    preds = []
    with torch.no_grad():
        for val in X_test:
            y_hat = model.forward(val)
            preds.append(y_hat.argmax().item())
    if was_training:
        model.train()
    return preds

if __name__ == '__main__':
    X, y = load_iris(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    model = NewModel()

    model.load_state_dict(torch.load('modelstate.pt'))

    y_pred = model_predict(model, X_test)
    print('accuracy = ', accuracy_score(y_test, y_pred))
