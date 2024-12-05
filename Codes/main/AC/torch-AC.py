import sys
sys.path.insert(0, '../../Utilities/')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import random

# 设置随机种子
torch.manual_seed(1234)
np.random.seed(1234)



class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, x1, layers, dt, lb, ub, q, device):
        super(PhysicsInformedNN, self).__init__()

        self.device = device
        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)
        self.dt = torch.tensor(dt, dtype=torch.float32).to(self.device)

        self.q = max(q, 1)

        # 初始化数据到设备
        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(self.device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(self.device)

        self.u0 = torch.tensor(u0, dtype=torch.float32).to(self.device)

        # 加载 IRK 权重并确保它们是 PyTorch 张量
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2))
        self.IRK_weights = torch.tensor(np.reshape(tmp[0:q ** 2 + q], (q + 1, q)), dtype=torch.float32).to(self.device)
        self.IRK_times = torch.tensor(tmp[q ** 2 + q:], dtype=torch.float32).to(self.device)

        # 构建网络层
        self.model = self.build_model(layers).to(self.device)
        self.optimizer_adam = optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(),
            max_iter=500, tolerance_grad=1e-5, tolerance_change=1e-9, line_search_fn="strong_wolfe"
        )

    def build_model(self, layers):
        model = nn.Sequential()
        for i in range(len(layers) - 2):
            model.add_module(f'layer_{i}', nn.Linear(layers[i], layers[i + 1]))
            model.add_module(f'activation_{i}', nn.Tanh())
        model.add_module(f'output_layer', nn.Linear(layers[-2], layers[-1]))

        # Xavier 初始化
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        return model

    def forward(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(H)

    def fwd_gradients(self, U, x):
        U_x = torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), create_graph=True)[0]
        U_xx = torch.autograd.grad(U_x, x, grad_outputs=torch.ones_like(U_x), create_graph=True)[0]
        return U_x, U_xx

    def net_U0(self, x):
        x.requires_grad_(True)
        U1 = self.forward(x)
        U = U1[:, :-1]
        U_x, U_xx = self.fwd_gradients(U, x)
        F = 5.0*U-5.0*U**3 + 0.0001*U_xx

        U0 = U1 - self.dt * torch.matmul(F, self.IRK_weights.T)
        return U0

    def net_U1(self, x):
        x.requires_grad_(True)
        U1 = self.forward(x)
        U1_x = self.fwd_gradients(U1, x)[0]
        return U1, U1_x

    def compute_loss(self):
        self.U0_pred = self.net_U0(self.x0)
        self.U1_pred, U1_x_pred = self.net_U1(self.x1)

        loss = torch.sum(torch.square(self.u0 - self.U0_pred)) + \
                torch.sum(torch.square(self.U1_pred[0, :] - self.U1_pred[1, :])) + \
                torch.sum(torch.square(U1_x_pred[0, :] - U1_x_pred[1, :]))
        return loss

    def train(self, nIter_adam, nIter_lbfgs):
        start_time = time.time()
        print("Starting training with Adam optimizer...")
        for epoch in range(nIter_adam):
            self.optimizer_adam.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer_adam.step()

            if epoch % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch(Adam): {epoch}, Loss: {loss.item():.5e}, Time: {elapsed:.2f}")
                start_time = time.time()

        print("Switching to L-BFGS optimizer...")

        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            return loss

        self.optimizer_lbfgs.step(closure)

    def predict(self, x_star):
        x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device)
        U1_star, _ = self.net_U1(x_star)
        return U1_star.detach().cpu().numpy()


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    q = 100
    layers = [1, 200, 200, 200, 200, 200, 200, q + 1]
    lb = np.array([-1.0])
    ub = np.array([1.0])

    N = 200

    data = scipy.io.loadmat('../Data/AC.mat')

    t = data['tt'].flatten()[:, None]  # T x 1
    x = data['x'].flatten()[:, None]  # N x 1
    Exact = np.real(data['uu']).T  # T x N

    idx_t0 = 20
    idx_t1 = 180
    dt = t[idx_t1] - t[idx_t0]

    # 初始数据
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    # 边界数据
    x1 = np.vstack((lb, ub))

    # 测试数据
    x_star = x

    model = PhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q, device)
    model.train(nIter_adam=50000, nIter_lbfgs=5000)

    U1_pred = model.predict(x_star)

    error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    print(f'Error: {error:.5e}')
    fig = plt.figure(figsize=(10, 12))

    # 关闭坐标轴
    fig.patch.set_facecolor('white')

    # ############################## Row 0: h(t,x) ##############################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    # 画出 Exact 的图像
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                  origin='lower', aspect='auto')

    # 添加色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # 添加时间 t0 和 t1 的垂直线
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[idx_t0] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[idx_t1] * np.ones((2, 1)), line, 'w-', linewidth=1)

    # 设置轴标签和标题
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize=10)

    # ############################## Row 1: h(t,x) slices ##############################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 1 / 2 - 0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    # 第一个子图，t = t0 的时刻
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[idx_t0, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x0, u0, 'rx', linewidth=2, label='Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$' % (t[idx_t0].item()), fontsize=10)
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)

    # 第二个子图，t = t1 的时刻
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_star, U1_pred[:, -1], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$' % (t[idx_t1].item()), fontsize=10)
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    plt.show()
'''
    # 可视化预测结果
    fig = plt.figure(figsize=(12, 10))
    # 第一个图
    gs0 = gridspec.GridSpec(1, 2)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cax=ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
              extent=[t.min(), t.max(), x_star.min(), x_star.max()],
              origin='lower', aspect='auto')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    plt.title('Prediction')
    plt.colorbar(cax, ax=ax)
    plt.show()
'''
