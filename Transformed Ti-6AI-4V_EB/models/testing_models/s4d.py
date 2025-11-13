import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from opt_einsum import contract
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print('WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.')
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class DropoutNd(nn.Module):

    def __init__(self, p: float=0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError('dropout probability has to be in [0, 1), but got {}'.format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X

class modrelu(nn.Module):

    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)
        return phase * magnitude

class Modrelu(modrelu):

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

class S4DKernel(nn.Module):

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kernel_args):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.H = H
        self.N = N // 2
        C = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        B = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.B = nn.Parameter(_c2r(B))
        self.register('log_dt', log_dt, lr)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register('log_A_real', log_A_real, lr)
        self.register('A_imag', A_imag, lr)

    def forward(self, L):
        C = _r2c(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = torch.exp(self.log_dt)
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {'weight_decay': 0.0}
            if lr is not None:
                optim['lr'] = lr
            setattr(getattr(self, name), '_optim', optim)

    def step(self, u, state):
        C = _r2c(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = torch.exp(self.log_dt)
        dtA = A * dt.unsqueeze(-1)
        self.dA = torch.exp(dtA)
        self.dC = C
        self.dB = self.dC.new_ones(self.H, self.N) * (torch.exp(dtA) - 1.0) / A
        next_state = contract('h n, b h n -> b h n', self.dA, state) + contract('h n, b h -> b h n', self.dB, u)
        y = contract('h n, b h n -> b h', self.dC, next_state)
        return (2 * y.real, next_state)

    def default_state(self, *batch_shape, device='cuda'):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=device)
        return state

class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, activation='gelu', bidirectional=False, postact='glu', **kernel_args):
        super().__init__()
        assert not bidirectional, 'not implemented yet'
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = Activation(activation)
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        if postact == 'glu':
            self.output_linear = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))
        else:
            self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return (y, None)

    def step(self, u, state, **kwargs):
        y, next_state = self.kernel.step(u, state)
        y = y + contract('bh,h->bh', u, self.D)
        y = self.activation(y)
        y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        return (y, next_state)

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)

class S4DJointKernel(nn.Module):

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kernel_args):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.H = H
        self.N = N // 2
        C = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        C_aux = torch.ones(H, N // 2, dtype=torch.cfloat)
        self.C_aux = nn.Parameter(_c2r(C_aux))
        E = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.E = nn.Parameter(_c2r(E))
        self.register('log_dt', log_dt, lr)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register('log_A_real', log_A_real, lr)
        self.register('A_imag', A_imag, lr)

    def forward(self, L):
        C = _r2c(self.C)
        C_aux = _r2c(self.C_aux)
        E = _r2c(self.E)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = torch.exp(self.log_dt)
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C_main = C * (torch.exp(dtA) - 1.0) / A
        C_aux = C_aux * (torch.exp(dtA) - 1.0) / A
        E_main = E * C_main
        E_aux = E * C_aux
        Ku = 2 * torch.einsum('hn, hnl -> hl', C_main, torch.exp(K)).real
        Kx = 2 * torch.einsum('hn, hnl -> hl', E_main, torch.exp(K)).real
        Ku_aux = 2 * torch.einsum('hn, hnl -> hl', C_aux, torch.exp(K)).real
        Kx_aux = 2 * torch.einsum('hn, hnl -> hl', E_aux, torch.exp(K)).real
        return (Ku, Kx, Ku_aux, Kx_aux)

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {'weight_decay': 0.0}
            if lr is not None:
                optim['lr'] = lr
            setattr(getattr(self, name), '_optim', optim)

    def step(self, u, x, state, **kwargs):
        C = _r2c(self.C)
        C_aux = _r2c(self.C_aux)
        E = _r2c(self.E)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = torch.exp(self.log_dt)
        dtA = A * dt.unsqueeze(-1)
        self.dA = torch.exp(dtA)
        self.dC_main = C
        self.dC_aux = C_aux
        self.dB = self.dC_main.new_ones(self.H, self.N) * (torch.exp(dtA) - 1.0) / A
        self.dE = E * (torch.exp(dtA) - 1.0) / A
        next_state = contract('h n, b h n -> b h n', self.dA, state) + contract('h n, b h -> b h n', self.dB, u) + contract('h n, b h -> b h n', self.dE, x)
        y = contract('h n, b h n -> b h', self.dC_main, next_state)
        y_aux = contract('h n, b h n -> b h', self.dC_aux, next_state)
        return (2 * y.real, 2 * y_aux.real, next_state)

    def default_state(self, *batch_shape, device='cuda'):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=device)
        return state

class S4DJoint(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, activation='gelu', bidirectional=False, postact='glu', **kernel_args):
        super().__init__()
        assert not bidirectional, 'not implemented yet'
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.D_main = nn.Parameter(torch.randn(self.h))
        self.F_main = nn.Parameter(torch.randn(self.h))
        self.D_aux = nn.Parameter(torch.randn(self.h))
        self.F_aux = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DJointKernel(self.h, N=self.n, **kernel_args)
        self.activation = Activation(activation)
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        if postact == 'glu':
            self.output_linear = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))
        else:
            self.output_linear = nn.Conv1d(self.h, self.h, kernel_size=1)
        if postact == 'glu':
            self.output_linear_aux = nn.Sequential(nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))
        else:
            self.output_linear_aux = nn.Conv1d(self.h, self.h, kernel_size=1)

    def forward(self, ins, t=None, **kwargs):
        u, x = ins
        if not self.transposed:
            x = x.transpose(-1, -2)
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        ku, kx, ku_aux, kx_aux = self.kernel(L=L)
        ku_f = torch.fft.rfft(ku, n=2 * L)
        kx_f = torch.fft.rfft(kx, n=2 * L)
        ku_aux_f = torch.fft.rfft(ku_aux, n=2 * L)
        kx_aux_f = torch.fft.rfft(kx_aux, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        x_f = torch.fft.rfft(x, n=2 * L)
        yu_main = torch.fft.irfft(u_f * ku_f, n=2 * L)[..., :L]
        yx_main = torch.fft.irfft(x_f * kx_f, n=2 * L)[..., :L]
        yu_aux = torch.fft.irfft(u_f * ku_aux_f, n=2 * L)[..., :L]
        yx_aux = torch.fft.irfft(x_f * kx_aux_f, n=2 * L)[..., :L]
        y_main = yu_main + yx_main + u * self.D_main.unsqueeze(-1) + x * self.F_main.unsqueeze(-1)
        y_aux = yu_aux + yx_aux + u * self.D_aux.unsqueeze(-1) + x * self.F_aux.unsqueeze(-1)
        y_main = self.dropout(self.activation(y_main))
        y_main = self.output_linear(y_main)
        y_aux = self.dropout(self.activation(y_aux))
        y_aux = self.output_linear_aux(y_aux)
        if not self.transposed:
            y_main, y_aux = (y_main.transpose(-1, -2), y_aux.transpose(-1, -2))
        return ((y_main, y_aux), None)

    def step(self, ins, state, **kwargs):
        u, x = ins
        y, y_aux, next_state = self.kernel.step(u, x, state)
        y = y + contract('bh,h->bh', u, self.D_main) + contract('bh,h->bh', x, self.F_main)
        y_aux = y_aux + contract('bh,h->bh', u, self.D_aux) + contract('bh,h->bh', x, self.F_aux)
        y = self.activation(y)
        y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        y_aux = self.activation(y_aux)
        y_aux = self.output_linear_aux(y_aux.unsqueeze(-1)).squeeze(-1)
        return ((y, y_aux), next_state)

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)

def test_step():
    device = 'cuda'
    B = 2
    H = 1
    N = 4
    L = 784
    s4 = S4DJoint(d_state=N, d_model=10)
    s4.to(device)
    for module in s4.modules():
        if hasattr(module, 'setup_step'):
            module.setup_step()
    u = torch.rand(B, H, L).to(device)
    x = torch.rand(B, H, L).to(device)
    initial_state = s4.default_state(B, device=device)
    y, yx, _ = s4((u, x))
    print('output mean:\n', y, y.shape)
    print('output std:\n', yx, yx.shape)
    state = initial_state
    ys = []
    yxs = []
    for i, u_ in enumerate(torch.unbind(u, dim=-1)):
        y_, yx_, state = s4.step((u_, x[..., i]), state=state)
        ys.append(y_)
        yxs.append(yx_)
    ys = torch.stack(ys, dim=-1)
    yxs = torch.stack(yxs, dim=-1)
    print('step outputs y:\n', ys, ys.shape)
    print('step outputs yx:\n', yx, yxs.shape)
    breakpoint()
if __name__ == '__main__':
    test_step()