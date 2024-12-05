import sys
sys.path.insert(0, '../../Utilities/')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import matplotlib.pyplot as plt
import time
import psutil
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 设置随机种子
torch.manual_seed(1234)
np.random.seed(1234)


class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, x1, u1, layers, dt, lb, ub, q, device):
        super(PhysicsInformedNN, self).__init__()

        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        self.x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True).to(device)
        self.x1 = torch.tensor(x1, dtype=torch.float32, requires_grad=True).to(device)

        self.u0 = torch.tensor(u0, dtype=torch.float32).to(device)
        self.u1 = torch.tensor(u1, dtype=torch.float32).to(device)

        self.dt = dt
        self.q = max(q, 1)

        # 初始化网络
        self.model = self.build_model(layers).to(device)

        # 初始化可训练参数
        self.lambda_1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32).to(device))
        self.lambda_2 = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32).to(device))

        # 加载IRK权重
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2))
        weights = np.reshape(tmp[0:q ** 2 + q], (q + 1, q))
        self.IRK_alpha = torch.tensor(weights[0:-1, :], dtype=torch.float32).to(device)
        self.IRK_beta = torch.tensor(weights[-1:, :], dtype=torch.float32).to(device)
        self.IRK_times = torch.tensor(tmp[q ** 2 + q:], dtype=torch.float32).to(device)

        self.device = device
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def build_model(self, layers):
        layers_list = []
        for i in range(len(layers) - 2):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            layers_list.append(nn.Tanh())
        layers_list.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*layers_list)

    def forward(self, X):
        # 数据标准化处理
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(X)

    def fwd_gradients(self, U, x):
        grad_u = torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), retain_graph=True, create_graph=True)[0]
        return \
        torch.autograd.grad(grad_u, x, grad_outputs=torch.ones_like(grad_u), retain_graph=True, create_graph=True)[0]

    def net_U0(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)

        U = self.forward(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)

        F = -lambda_1 * U * U_x - lambda_2 * U_xxx
        U0 = U - self.dt * torch.matmul(F, self.IRK_alpha.T)
        return U0

    def net_U1(self, x):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)

        U = self.forward(x)
        U_x = self.fwd_gradients(U, x)
        U_xx = self.fwd_gradients(U_x, x)
        U_xxx = self.fwd_gradients(U_xx, x)

        F = -lambda_1 * U * U_x - lambda_2 * U_xxx
        U1 = U + self.dt * torch.matmul(F, (self.IRK_beta - self.IRK_alpha).T)
        return U1

    def compute_loss(self):
        U0_pred = self.net_U0(self.x0)
        U1_pred = self.net_U1(self.x1)

        loss = torch.mean((self.u0 - U0_pred) ** 2) + torch.mean((self.u1 - U1_pred) ** 2)
        return loss

    def train_model(self, nIter):
        for it in range(nIter):
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()

            if it % 10 == 0:
                lambda_1_value = self.lambda_1.item()
                lambda_2_value = torch.exp(self.lambda_2).item()
                print(f"It: {it}, Loss: {loss.item():.3e}, l1: {lambda_1_value:.3f}, l2: {lambda_2_value:.5f}")

    def predict(self, x_star):
        x_star = torch.tensor(x_star, dtype=torch.float32, requires_grad=True).to(self.device)

        U0_star = self.net_U0(x_star).detach().cpu().numpy()
        U1_star = self.net_U1(x_star).detach().cpu().numpy()

        return U0_star, U1_star


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = 50
    skip = 120

    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]

    data = scipy.io.loadmat('../Data/KdV.mat')

    t_star = data['tt'].flatten()[:, None]
    x_star = data['x'].flatten()[:, None]
    Exact = np.real(data['uu'])

    idx_t = 40
    noise = 0.0

    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x, :]
    u0 = Exact[idx_x, idx_t][:, None] + noise * np.std(Exact[idx_x, idx_t]) * np.random.randn(N0, 1)

    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x, :]
    u1 = Exact[idx_x, idx_t + skip][:, None] + noise * np.std(Exact[idx_x, idx_t + skip]) * np.random.randn(N1, 1)

    dt = (t_star[idx_t + skip] - t_star[idx_t]).item()
    lb = x_star.min(0)
    ub = x_star.max(0)

    model = PhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q, device)
    model.to(device)
    model.train_model(nIter=500)

    U0_pred, U1_pred = model.predict(x_star)

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = torch.exp(model.lambda_2).item()

    error_lambda_1 = abs(lambda_1_value - 1.0) / 1.0 * 100
    error_lambda_2 = abs(lambda_2_value - 0.0025) / 0.0025 * 100

    print(f"Error lambda_1: {error_lambda_1:.5f}%")
    print(f"Error lambda_2: {error_lambda_2:.5f}%")


# 假设 H_pred, Exact_h, X, T, x, t, lb, ub 已经定义
#fig = plt.figure(figsize=(10, 8))
# 创建图形和布局
fig = plt.figure(figsize=(12, 10))
gs0 = gridspec.GridSpec(1, 2, top=1 - 0.06, bottom=1 - 1/3 + 0.05, left=0.15, right=0.85, wspace=0)
ax_heatmap = fig.add_subplot(gs0[:, :])

# 第一部分：二维热图
h = ax_heatmap.imshow(
    Exact, interpolation='nearest', cmap='rainbow',
    extent=[t_star.min(), t_star.max(), lb[0], ub[0]],
    origin='lower', aspect='auto'
)
divider = make_axes_locatable(ax_heatmap)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax_heatmap.set_xlabel('$t$')
ax_heatmap.set_ylabel('$x$')
ax_heatmap.set_title('$u(t,x)$', fontsize=10)

# 绘制时间轴上的白线
line = np.linspace(x_star.min(), x_star.max(), 2)[:, None]
ax_heatmap.plot(t_star[idx_t] * np.ones((2, 1)), line, 'w-', linewidth=1.0)
ax_heatmap.plot(t_star[idx_t + skip] * np.ones((2, 1)), line, 'w-', linewidth=1.0)

# 第二部分：不同时间点的曲线图
gs1 = gridspec.GridSpec(1, 2, top=1 - 1/3 - 0.1, bottom=1 - 2/3, left=0.15, right=0.85, wspace=0.5)

# t = idx_t 时间切片的曲线图
ax1 = fig.add_subplot(gs1[0, 0])
ax1.plot(x_star, Exact[:, idx_t], 'b', linewidth=2, label='Exact')
ax1.plot(x0, u0.cpu().numpy() if isinstance(u0, torch.Tensor) else u0, 'rx', linewidth=2, label='Data')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$u(t,x)$')
ax1.set_title('$t = %.2f$\n%d training data' % (t_star[idx_t].item(), u0.shape[0]), fontsize=10)

# t = idx_t + skip 时间切片的曲线图
ax2 = fig.add_subplot(gs1[0, 1])
ax2.plot(x_star, Exact[:, idx_t + skip], 'b', linewidth=2, label='Exact')
ax2.plot(x1, u1.cpu().numpy() if isinstance(u1, torch.Tensor) else u1, 'rx', linewidth=2, label='Data')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$u(t,x)$')
ax2.set_title('$t = %.2f$\n%d training data' % (t_star[idx_t + skip].item(), u1.shape[0]), fontsize=10)
ax2.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

# 第三部分：PDE结构标注信息
gs2 = gridspec.GridSpec(1, 2, top=1 - 2/3 - 0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)

ax_text = fig.add_subplot(gs2[0, 0])
ax_text.axis('off')

# 使用多行字符串格式化输出，替换 LaTeX 表格
s = (
    r"Correct PDE: $u_t + u u_x + 0.0025 u_{xxx} = 0$" "\n"
    r"Identified PDE (clean data): "
    r"$u_t + %.3f u u_x + %.7f u_{xxx} = 0$" "\n"
    r"Identified PDE (1%% noise): "
    r"$u_t + %.3f u u_x + %.7f u_{xxx} = 0$"
    % (lambda_1_value, lambda_2_value, error_lambda_1, error_lambda_2)
)
ax_text.text(0, 0.5, s, fontsize=10, ha='left')

# 保存并显示图像
fig.savefig("output_image.png", dpi=300)
plt.show()

