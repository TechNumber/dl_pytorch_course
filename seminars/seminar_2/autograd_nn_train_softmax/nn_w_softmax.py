import numpy as np

import autograd_w_softmax

class Module:

    def zero_grad(self):
        self.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [autograd_w_softmax.Value(np.random.rand()) for i in range(nin)]
        self.b = autograd_w_softmax.Value(np.random.rand())
        self.nonlin = nonlin

    def __call__(self, x):
        batch_s = len(x)
        nin = len(x[0])
        res = [self.b for i in range(batch_s)]
        for i in range(batch_s):
            for j in range(nin):
                res[i] += self.w[j] * x[i][j]
        return [r.relu() for r in res] if self.nonlin else res

    def zero_grad(self):
        for weight in self.w:
            weight.grad = 0
        self.b.grad = 0

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, kwargs['nonlin']) for i in range(nout)]
        self.nonlin = kwargs['nonlin']

    def __call__(self, x):
        out = [[] for i in range(len(x))]
        for n in self.neurons:
            n_out = n(x)
            for i in range(len(x)):
                out[i].append(n_out[i])
        if not self.nonlin:
            out_temp = [[0] * len(out[0]) for i in range(len(out))]
            for i in range(len(out)):
                for j in range(len(out[0])):
                    out_temp[i][j] = out[i][j].softmax(*[out[i][k] for k in range(len(out[0])) if k != j])
        else:
            out_temp = out
        return out_temp[0] if len(out_temp) == 1 else out_temp

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    def parameters(self):
        l_params = []
        for n in self.neurons:
            l_params.extend(n.parameters())
        return l_params

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    def parameters(self):
        nn_params = []
        for l in self.layers:
            nn_params.extend(l.parameters())
        return nn_params

    def __repr__(self):
        repr = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of [{repr}]"
