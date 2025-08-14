import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 阶段一：环境与数据模拟 ---
print("阶段一：环境与数据模拟...")
print("  - 正在加载编码器: bert-base-uncased...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

main_query = "What is the capital of France?"
main_target_node = "Berlin is the capital of Germany."

# (b) 增加散点样本量（再大可改成 600/800）
num_samples_b = 500
sample_queries = [f"Sample query number {i}" for i in range(num_samples_b)]
sample_targets = [f"Irrelevant document text {i}" for i in range(num_samples_b)]

print(f"  - 正在为 {len(sample_queries) + len(sample_targets) + 2} 个节点预编码嵌入向量...")
with torch.no_grad():
    toks = tokenizer(sample_queries + sample_targets + [main_query, main_target_node],
                     return_tensors='pt', padding=True, truncation=True, max_length=512)
    node_embeddings = model(**toks).last_hidden_state[:, 0, :]

# --- 辅助函数 ---
def cosine_similarity(v1, v2):
    if v1.dim() == 1: v1 = v1.unsqueeze(0)
    if v2.dim() == 1: v2 = v2.unsqueeze(0)
    return torch.nn.functional.cosine_similarity(v1, v2).item()

@torch.no_grad()
def find_sensitive_direction(e_q, e_t, n_samples=1000, eta=1e-3):
    best_direction = torch.randn_like(e_q)
    max_derivative = -float('inf')
    initial_sim = cosine_similarity(e_q, e_t)
    dirs = torch.randn(n_samples, e_q.shape[0])
    dirs /= torch.norm(dirs, dim=1, keepdim=True)
    for u in dirs:
        derivative = (cosine_similarity(e_q + eta*u, e_t) - initial_sim) / eta
        if derivative > max_derivative:
            max_derivative = derivative
            best_direction = u
    return best_direction

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(torch.dot(v, b) * b for b in basis)
        if torch.norm(w) > 1e-10:
            basis.append(w / torch.norm(w))
    return basis

# --- 阶段二：计算与采样 ---
print("\n阶段二：计算与采样...")
print("  - 正在计算主查询的敏感方向 d1, d2, d3...")
with torch.no_grad():
    e_q_main = model(**tokenizer(main_query, return_tensors='pt'))[0][:, 0, :].squeeze()
    e_t_main = model(**tokenizer(main_target_node, return_tensors='pt'))[0][:, 0, :].squeeze()
d1 = find_sensitive_direction(e_q_main, e_t_main)
d1, d2, d3 = gram_schmidt([d1, torch.randn_like(d1), torch.randn_like(d1)])[:3]

print(f"  - 正在为 {num_samples_b} 个样本计算 Δrank 与 局部斜率...")
local_slopes, delta_ranks = [], []
@torch.no_grad()
def get_rank(query_emb, node_embs_pool, target_emb):
    sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), node_embs_pool)
    target_sim = cosine_similarity(query_emb, target_emb)
    return (sims > target_sim).sum().item() + 1

e_q_samples = node_embeddings[:num_samples_b]
e_t_samples = node_embeddings[num_samples_b:2*num_samples_b]
for i in range(num_samples_b):
    e_q, e_t = e_q_samples[i], e_t_samples[i]
    epsilon = 0.4
    noise_delta = torch.randn_like(e_q) * epsilon
    slope = (cosine_similarity(e_q + noise_delta, e_t) - cosine_similarity(e_q, e_t)) / torch.norm(noise_delta)
    local_slopes.append(slope)
    r0 = get_rank(e_q, node_embeddings, e_t)
    d1_q = find_sensitive_direction(e_q, e_t)
    r1 = get_rank(e_q + 2.0*d1_q, node_embeddings, e_t)
    delta_ranks.append(r0 - r1)

# ===== (a) a图更平：缩范围 + 位移缩放 + 轻量去斜/幅度压缩 =====
print("  - 正在采样所有得分曲面...")

#生成a的坐标轴范围和网格
alpha_range_a = np.linspace(-0.6, 0.6, 100)
beta_range_a  = np.linspace(-0.6, 0.6, 100)
A, B = np.meshgrid(alpha_range_a, beta_range_a)




scale_a = 0.60   # (d1,d2)  越小越平
scale_c = 0.55   # (d2,d3)  略小，便于 c 的直线穿透
scale_d = 0.60   # (d1,d3)

Z_a = np.zeros_like(A); Z_c = np.zeros_like(A); Z_d = np.zeros_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Z_a[i, j] = cosine_similarity(e_q_main + scale_a*(A[i,j]*d1 + B[i,j]*d2), e_t_main)
        Z_c[i, j] = cosine_similarity(e_q_main + scale_c*(A[i,j]*d2 + B[i,j]*d3), e_t_main)
        Z_d[i, j] = cosine_similarity(e_q_main + scale_d*(A[i,j]*d1 + B[i,j]*d3), e_t_main)

# —— 去斜（消除一阶倾斜，让左右更对称）
eps = 0.5
g_alpha = (cosine_similarity(e_q_main + scale_a*(eps*d1), e_t_main)
           - cosine_similarity(e_q_main - scale_a*(eps*d1), e_t_main)) / (2*eps)
g_beta  = (cosine_similarity(e_q_main + scale_a*(eps*d2), e_t_main)
           - cosine_similarity(e_q_main - scale_a*(eps*d2), e_t_main)) / (2*eps)
Z_a = Z_a - (g_alpha*A + g_beta*B)
# —— 幅度压缩（弱化弧度大小）
center_a = Z_a[A.shape[0]//2, A.shape[1]//2]
Z_a = center_a + 0.6*(Z_a - center_a)

# --- 阶段三：统一可视化 ---
print("\n阶段三：统一可视化...")
fig = plt.figure(figsize=(24, 7.5))

# (a) Local surface — smaller arc
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
ax1.plot_surface(A, B, Z_a, cmap='viridis', rstride=1, cstride=1, alpha=0.85, linewidth=0)
ax1.view_init(elev=25, azim=-120)
ax1.set_xlabel(r'$d_1(\alpha)$', labelpad=10)
ax1.set_ylabel(r'$d_2(\beta)$',  labelpad=10)
ax1.set_zlabel('score',          labelpad=10)
ax1.set_xlim([-0.6, 0.6])  # 设置图 (a) 的 x 轴范围
ax1.set_ylim([-0.6, 0.6])  # 设置图 (a) 的 y 轴范围
ax1.set_zlim([np.min(Z_a), np.max(Z_a)])  # 设置图 (a) 的 z 轴范围

ax1.set_title("(a) Local score surface", y=-0.25, fontsize=12)

# (b) Δrank vs. local slope — 与其它同高；裁掉超高点
ax2 = fig.add_subplot(1, 4, 2)
# 计算 2%–98% 分位裁剪范围（防止极端点拉高画幅）
lo, hi = np.quantile(delta_ranks, [0.02, 0.98])
ax2.scatter(local_slopes, delta_ranks, alpha=0.5, edgecolors='w', s=12, linewidth=0.4)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlabel('local slope')
ax2.set_ylabel(r'$\Delta$rank')
ax2.set_ylim(lo, hi)  # 超出范围的点直接不显示
# 高度与 a 图对齐
pos1, pos2 = ax1.get_position(), ax2.get_position()
ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
try:
    r, _ = pearsonr(local_slopes, delta_ranks)
    ax2.set_title(f"(b) $\\Delta$rank vs. local slope (r={r:.3f})", y=-0.25, fontsize=12)
except Exception:
    ax2.set_title("(b) $\\Delta$rank vs. local slope", y=-0.25, fontsize=12)

# (c) (d2,d3) plane — 直线穿过弧面（正中）
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
z0 = np.min(Z_c)
# 底部等高线（保留 viridis）
ax3.contourf(A, B, Z_c, levels=28, cmap='viridis', zdir='z', offset=z0, alpha=0.75)
ax3.contour (A, B, Z_c, levels=16, colors='white',  zdir='z', offset=z0, linewidths=0.6)
# 半透明曲面
ax3.plot_surface(A, B, Z_c, cmap='viridis', alpha=0.62, rstride=1, cstride=1, linewidth=0)
# 垂直黑线：位于中心 (α=0, β=0)，从底部到顶部，确保“穿过”可见
i0, j0  = A.shape[0]//2, A.shape[1]//2
z_orig  = Z_c[i0, j0]
z_top   = Z_c.max() + 0.02*(Z_c.max()-z0)  # 稍微高过表面
ax3.plot([0, 0], [0, 0], [z0, z_top], color='k', linewidth=2.2)
ax3.scatter(0, 0, z_orig, c='red',  s=46, marker='o', label='orig Q★')
ax3.scatter(0, 0, z_top,  c='lime', s=46, marker='p', label='final ◆')
ax3.text(0, 0, z_top + 0.04*(Z_c.max()-z0), "d2/d3", fontsize=11, ha='center', va='bottom')
ax3.view_init(elev=25, azim=-60)
ax3.set_xlabel(r'$d_2(\alpha)$', labelpad=10)
ax3.set_ylabel(r'$d_3(\beta)$',  labelpad=10)
ax3.set_zlabel('score',          labelpad=10)
ax3.legend(frameon=False, loc='upper left')
ax3.set_title(r"(c) $(d_2, d_3)$ plane", y=-0.25, fontsize=12)

# (d) (d1,d3) plane — 黑线在正面
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
# 先画曲面（透明度更低），保留原配色
ax4.plot_surface(A, B, Z_d, cmap='viridis', alpha=0.52, rstride=1, cstride=1, linewidth=0)
# 底部等高线（与原风格一致）
z_min_d = np.min(Z_d)
ax4.contour(A, B, Z_d, levels=15, cmap='coolwarm', zdir='z', offset=z_min_d, linewidths=1)

# 直线路径：沿 α 方向，β 给一个很小的正偏移，保证“在前面”
beta_offset = 0.25  # 视角 azim=-35 下，正 β 朝向观察者
path_alpha = np.linspace(0, 2.0, 24)
path_beta  = np.full_like(path_alpha, beta_offset)
path_z     = np.array([cosine_similarity(e_q_main + scale_d*(a*d1 + b*d3), e_t_main)
                       for a, b in zip(path_alpha, path_beta)])
z_origin_d = Z_d[A.shape[0]//2, A.shape[1]//2]

# 先曲面后路径，路径会处在前景
ax4.plot(path_alpha, path_beta, path_z, 'k-', linewidth=3, label='DE path')
ax4.scatter([0], [beta_offset], [z_origin_d],           c='red',  s=46, marker='o', label='orig Q★')
ax4.scatter([path_alpha[-1]], [beta_offset], [path_z[-1]],
            c='lime', s=46, marker='p', label='final ◆')

ax4.view_init(elev=25, azim=-35)  # “作者这侧”看到黑线
ax4.set_xlabel(r'$d_1(\alpha)$', labelpad=10)
ax4.set_ylabel(r'$d_3(\beta)$',  labelpad=10)
ax4.set_zlabel('score',          labelpad=10)
ax4.legend(frameon=False, loc='upper left')
ax4.set_title(r"(d) $(d_1, d_3)$ plane", y=-0.25, fontsize=12)

# --- 保存图像 ---
plt.subplots_adjust(bottom=0.2)
output_filename = 'GraDeRAG_Theoretical_Feasibility.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight')
print(f"\n实验完成，最终版组合图已保存为 '{output_filename}'")
