import autograd
import nn
import random
import numpy as np
import torch
from pytorch_lightning import seed_everything


def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    seed_everything(s, workers=True)


SEED = 17
set_random_seed(SEED)

model = nn.MLP(3, [4, 4, 1])
# 1-й слой: 3 входящих признака, 4 нейрона (4 исходящих признака)
# 2-й слой: 4 входящих признака, 4 нейрона (4 исходящих признака)
# 3-й слой: 4 входящих признака, 1 нейрон (1 исходящий признак)
print(model)
print("number of parameters", len(model.parameters()))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

learning_rate = 0.01
loss_hist = []
acc_hist = []
for k in range(150):
    model.zero_grad()

    # forward
    pred = model(xs)

    # calculate loss (mean square error)
    total_loss = autograd.Value(0)
    for i in range(len(ys)):
        total_loss += (pred[i][0] - ys[i]) ** 2
    total_loss /= len(xs)

    # backward (zero_grad + backward)
    total_loss.backward()
    loss_hist.append(total_loss.data)

    # accuracy
    mean_y = 0
    for y in ys:
        mean_y += y
    mean_y /= len(ys)
    var_y = 0
    for y in ys:
        var_y += (y - mean_y) ** 2
    var_y /= len(ys)
    acc = 1 - total_loss.data / var_y
    acc_hist.append(0 if acc < 0 else acc)

    # update
    # for layer_p in model.parameters():
    #     for neuron_p in layer_p:
    #         for w in neuron_p[0]:
    #             w.data -= w.grad * learning_rate
    #         neuron_p[1].data -= neuron_p[1].grad
    for p in model.parameters():
        p.data -= p.grad * learning_rate

    # logging
    if k % 1 == 0:
        print(f"step: {k}, loss: {round(total_loss.data, 6)}, accuracy R^2: {round(acc * 100, 2)}%")

print([round(p[0].data, 3) for p in model(xs)])
print(ys)
