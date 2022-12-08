import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def test(model, test_loader, device):
    model.eval()

    if model.beta_ema > 0:
        old_params = model.get_params()
        model.load_ema_params()

    pred = []
    ground_truth = []

    for test_u, test_i, test_ratings in test_loader:
        test_u, test_i, test_ratings = test_u.to(device), test_i.to(device), test_ratings.to(device)
        scores = model(test_u, test_i)
        pred.append(list(scores[0].data.cpu().numpy()))
        ground_truth.append(list(test_ratings.data.cpu().numpy()))

    pred = np.array(sum(pred, []), dtype = np.float32)
    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)

    rmse = sqrt(mean_squared_error(pred, ground_truth))
    mae = mean_absolute_error(pred, ground_truth)

    if model.beta_ema > 0:
        model.load_params(old_params)
    return rmse, mae