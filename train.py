from datetime import datetime


def train(model, train_loader, optimizer, epoch, rmse_mn, mae_mn, device):
    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        batch_u, batch_i, batch_ratings = data

        optimizer.zero_grad()

        loss = model.loss(batch_u.to(device), batch_i.to(device), batch_ratings.to(device).squeeze())
        loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss += loss.item()

        # clamp the parameters
        layers = model.layers
        for k, layer in enumerate(layers):
            layer.constrain_parameters()

        if model.beta_ema > 0.:
            model.update_ema()

        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f, the best RMSE/MAE: %.5f / %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10, rmse_mn, mae_mn))
            avg_loss = 0.0
    return 0
