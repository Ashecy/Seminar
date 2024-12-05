import sys

sys.path.insert(0, '../../Utilities/')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
torch.manual_seed(1234)


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return torch.tensor(np.random.randn(in_dim, out_dim) * xavier_stddev, dtype=torch.float32, requires_grad=True)


class PhysicsInformedNN:
    def __init__(self, x, y, t, u, v, layers):
        X = torch.cat([x, y, t], dim=1)

        self.lb = X.min(0).values
        self.ub = X.max(0).values

        self.X = X
        self.x = x.requires_grad_(True)
        self.y = y.requires_grad_(True)
        self.t = t.requires_grad_(True)
        self.u = u
        self.v = v

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        self.lambda_2 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=1e-3)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = xavier_init([layers[l], layers[l + 1]])
            b = torch.zeros((1, layers[l + 1]), dtype=torch.float32, requires_grad=True)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def get_parameters(self):
        params = [self.lambda_1, self.lambda_2]
        params += self.weights
        params += self.biases
        return params

    def neural_net(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(len(self.weights) - 1):
            H = torch.tanh(torch.add(torch.matmul(H, self.weights[l]), self.biases[l]))
        Y = torch.add(torch.matmul(H, self.weights[-1]), self.biases[-1])
        return Y

    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(torch.cat([x, y, t], dim=1))
        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        return u, v, p, f_u, f_v

    def train(self, nIter):
        for it in range(nIter):
            self.optimizer.zero_grad()
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean((self.v - v_pred) ** 2) + \
                   torch.mean(f_u_pred ** 2) + torch.mean(f_v_pred ** 2)
            loss.backward()
            self.optimizer.step()

            if it % 100 == 0:
                print(f'It: {it}, Loss: {loss.item():.3e}, l1: {self.lambda_1.item():.3f}, l2: {self.lambda_2.item():.5f}')

    def predict(self, x_star, y_star, t_star):
        x_star = x_star.requires_grad_(True)
        y_star = y_star.requires_grad_(True)
        t_star = t_star.requires_grad_(True)
        u_star, v_star, p_star, _, _ = self.net_NS(x_star, y_star, t_star)
        return u_star, v_star, p_star


def plot_solution(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.savefig(f'plot_{index}.png')  # Save the plot as an image file


if __name__ == "__main__":
    N_train = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = torch.tensor(XX.flatten()[:, None], dtype=torch.float32)  # NT x 1
    y = torch.tensor(YY.flatten()[:, None], dtype=torch.float32)  # NT x 1
    t = torch.tensor(TT.flatten()[:, None], dtype=torch.float32)  # NT x 1

    u = torch.tensor(UU.flatten()[:, None], dtype=torch.float32)  # NT x 1
    v = torch.tensor(VV.flatten()[:, None], dtype=torch.float32)  # NT x 1

    # Training Data
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(2000)

    # Test Data
    snap = np.array([100])
    x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
    y_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
    t_star = torch.tensor(TT[:, snap], dtype=torch.float32)

    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)

    # Error and plot
    u_star = UU[:, snap].flatten()
    plot_solution(X_star, u_pred.detach().numpy(), 1)

    # Plot vorticity data
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
    x_vort = data_vort['x']
    y_vort = data_vort['y']
    w_vort = data_vort['w']
    modes = np.squeeze(data_vort['modes'])
    nel = np.squeeze(data_vort['nel'])

    xx_vort = np.reshape(x_vort, (modes + 1, modes + 1, nel), order='F')
    yy_vort = np.reshape(y_vort, (modes + 1, modes + 1, nel), order='F')
    ww_vort = np.reshape(w_vort, (modes + 1, modes + 1, nel), order='F')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:, :, i], yy_vort[:, :, i], ww_vort[:, :, i], cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize=10)
    plt.savefig('plot_vorticity.png')

    # Plot predicted pressure vs exact pressure
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    PP_star = griddata(X_star, p_pred.detach().numpy().flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, P_star[:, snap].flatten(), (X, Y), method='cubic')

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    h = ax[0].imshow(PP_star, interpolation='nearest', cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$y$')
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title('Predicted pressure', fontsize=10)

    h = ax[1].imshow(P_exact, interpolation='nearest', cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$y$')
    ax[1].set_aspect('equal', 'box')
    ax[1].set_title('Exact pressure', fontsize=10)
    plt.savefig('plot_pressure_comparison.png')

    # Generate PDE comparison table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    correct_pde = (r'$u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})$',
                   r'$v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})$')

    identified_pde_clean = (r'$u_t + {:.3f} (u u_x + v u_y) = -p_x + {:.5f} (u_{{xx}} + u_{{yy}})$'.format(
                         model.lambda_1.item(), model.lambda_2.item()),
                         r'$v_t + {:.3f} (u v_x + v v_y) = -p_y + {:.5f} (v_{{xx}} + v_{{yy}})$'.format(
                         model.lambda_1.item(), model.lambda_2.item()))

    identified_pde_noisy = (r'$u_t + {:.3f} (u u_x + v u_y) = -p_x + {:.5f} (u_{{xx}} + u_{{yy}})$'.format(
                         model.lambda_1.item(), model.lambda_2.item()),
                         r'$v_t + {:.3f} (u v_x + v v_y) = -p_y + {:.5f} (v_{{xx}} + u_{{yy}})$'.format(
                         model.lambda_1.item(), model.lambda_2.item()))

    table_data = [['Correct PDE', correct_pde[0], correct_pde[1]],
                  ['Identified PDE (clean data)', identified_pde_clean[0], identified_pde_clean[1]],
                  ['Identified PDE (1% noise)', identified_pde_noisy[0], identified_pde_noisy[1]]]

    table = ax.table(cellText=table_data, colLabels=['Equation Type', 'Equation 1', 'Equation 2'], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(table_data[0]))))

    plt.savefig('pde_comparison_table.png')

    plt.show()

