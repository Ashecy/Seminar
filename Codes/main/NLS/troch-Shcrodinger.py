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
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
# 设置随机种子
# 设置随机种子
torch.manual_seed(1234)
np.random.seed(1234)


class PhysicsInformedNN(nn.Module):
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, device):
        super(PhysicsInformedNN, self).__init__()

        # 设备参数
        self.device = device

        # 将 lb 和 ub 移动到指定设备
        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)

        # 初始化数据并移动到指定设备
        self.x0 = x0.clone().detach().to(self.device).float()
        self.u0 = u0.clone().detach().to(self.device).float()
        self.v0 = v0.clone().detach().to(self.device).float()
        self.tb = tb.clone().detach().to(self.device).float()
        self.x_f = X_f[:, 0:1].clone().detach().to(self.device).float()
        self.t_f = X_f[:, 1:2].clone().detach().to(self.device).float()

        # 构建网络层
        self.model = self.build_model(layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)  # 确定学习率

    def build_model(self, layers):
        layers_list = []
        for i in range(len(layers) - 2):
            layers_list.append(nn.Linear(layers[i], layers[i + 1]))
            layers_list.append(nn.Tanh())
        layers_list.append(nn.Linear(layers[-2], layers[-1]))
        model = nn.Sequential(*layers_list)

        # Xavier 初始化
        for layer in model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        return model

    def forward(self, x, t):
        X = torch.cat([x, t], dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.model(X)
        u, v = uv[:, 0:1], uv[:, 1:2]
        return u, v

    def net_f_uv(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        u, v = self.forward(x, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u
        return f_u, f_v

    def compute_loss(self):
        u0_pred, v0_pred = self.forward(self.x0, self.tb)
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

        loss = torch.mean((self.u0 - u0_pred) ** 2) + \
               torch.mean((self.v0 - v0_pred) ** 2) + \
               torch.mean(f_u_pred ** 2) + \
               torch.mean(f_v_pred ** 2)
        return loss

    def train(self, nIter):
        start_time = time.time()
        max_gpu_memory = 0

        for epoch in range(nIter):
            epoch_start = time.time()
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()

            # 每轮迭代时间和内存
            epoch_end = time.time()
            cpu_memory = psutil.Process().memory_info().rss / 1024 ** 2  # MB
            gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
            max_gpu_memory = max(max_gpu_memory, gpu_memory)

            # 每100轮打印一次
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.5e}")
                print(
                    f"Epoch time: {epoch_end - epoch_start:.2f}s, CPU memory: {cpu_memory:.2f} MB, GPU memory: {gpu_memory:.2f} MB")

        # 总时间和最大GPU内存
        end_time = time.time()
        print(f"\nTotal Training time: {end_time - start_time:.2f} seconds")
        print(f"Peak GPU memory usage: {max_gpu_memory:.2f} MB")

    def predict(self, X_star):
        x_star = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(self.device)
        t_star = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(self.device)

        u_pred, v_pred = self.forward(x_star, t_star)
        f_u_pred, f_v_pred = self.net_f_uv(x_star, t_star)

        return u_pred.detach().cpu().numpy(), v_pred.detach().cpu().numpy(), f_u_pred.detach().cpu().numpy(), f_v_pred.detach().cpu().numpy()


# 加载数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = scipy.io.loadmat('../Data/NLS.mat')
t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = data['uu']
Exact_u = np.real(Exact)
Exact_v = np.imag(Exact)
Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()[:, None]
v_star = Exact_v.T.flatten()[:, None]
h_star = Exact_h.T.flatten()[:, None]

# 初始化模型
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi / 2])
N0 = 50
N_b = 50
N_f = 20000
layers = [2, 100, 100, 100, 100, 2]

# 采样边界数据
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = Exact_u[idx_x, 0:1]
v0 = Exact_v[idx_x, 0:1]

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

X_f = lb + (ub - lb) * lhs(2, N_f)

# 将 numpy 数组转换为 PyTorch 张量并移动到指定设备
x0 = torch.tensor(x0, dtype=torch.float32).to(device)
u0 = torch.tensor(u0, dtype=torch.float32).to(device)
v0 = torch.tensor(v0, dtype=torch.float32).to(device)
tb = torch.tensor(tb, dtype=torch.float32).to(device)
X_f = torch.tensor(X_f, dtype=torch.float32).to(device)

model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub, device).to(device)

# 训练模型 训练轮数
model.train(nIter=50000)

# 预测
u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

# 误差计算
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)

print(f'Error u: {error_u:.5e}')
print(f'Error v: {error_v:.5e}')
print(f'Error h: {error_h:.5e}')

# 可视化预测结果
H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
# 假设 H_pred, Exact_h, X, T, x, t, lb, ub 已经定义
fig = plt.figure(figsize=(10, 8))

# 使用 GridSpec 设置布局
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

# 第一张图：二维热图
h_img = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]], origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h_img, cax=cax)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$|h(t,x)|$', fontsize=10)

# 第二部分：不同时间点的曲线图
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

# t = 75 时间切片的曲线图
ax1 = plt.subplot(gs1[0, 0])
ax1.plot(x, Exact_h[:, 75], 'b-', linewidth=2, label='Exact')
ax1.plot(x, H_pred[75, :], 'r--', linewidth=2, label='Prediction')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$|h(t,x)|$')
ax1.set_title('$t = %.2f$' % t[75].item(), fontsize=10)
ax1.axis('square')
ax1.set_xlim([lb[0], ub[0]])
ax1.set_ylim([-0.1, 5.1])

# t = 100 时间切片的曲线图
ax2 = plt.subplot(gs1[0, 1])
ax2.plot(x, Exact_h[:, 100], 'b-', linewidth=2, label='Exact')
ax2.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|h(t,x)|$')
ax2.axis('square')
ax2.set_xlim([lb[0], ub[0]])
ax2.set_ylim([-0.1, 5.1])
ax2.set_title('$t = %.2f$' % t[100].item(), fontsize=10)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

# t = 125 时间切片的曲线图
ax3 = plt.subplot(gs1[0, 2])
ax3.plot(x, Exact_h[:, 125], 'b-', linewidth=2, label='Exact')
ax3.plot(x, H_pred[125, :], 'r--', linewidth=2, label='Prediction')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$|h(t,x)|$')
ax3.axis('square')
ax3.set_xlim([lb[0], ub[0]])
ax3.set_ylim([-0.1, 5.1])
ax3.set_title('$t = %.2f$' % t[125].item(), fontsize=10)

# 显示图像
#plt.tight_layout()
plt.show()
#fig.savefig("output_image.png", dpi=300)
fig.savefig("output_image.png", dpi=300, bbox_inches='tight')
'''
# 可视化
H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
fig, ax = plt.subplots()
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h_img = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]], origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h_img, cax=cax)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$|h(t,x)|$', fontsize=10)
plt.show()
fig.savefig("output_image.png")
from IPython.display import display
display(fig)

'''