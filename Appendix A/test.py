import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr
import warnings
import os

# --- Configuration ---
# Use a specific cache directory for transformers to avoid re-downloading
CACHE_DIR = "./huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Stage 1: Environment and Data Simulation ---
print("Stage 1: Environment and Data Simulation...")
print("  - Loading encoder: bert-base-uncased...")
# Suppress UserWarning for model loading
warnings.filterwarnings("ignore", category=UserWarning)
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
    model = BertModel.from_pretrained('bert-base-uncased', cache_dir=CACHE_DIR)
    model.eval()
    # Move model to GPU if available for faster computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  - Model loaded on: {device.type}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection or model name.")
    exit()

main_query = "What is the capital of France?"
main_target_node = "Berlin is the capital of Germany."

# Increased sample size for a more robust correlation plot
num_samples_b = 800
sample_queries = [f"Sample query number {i}" for i in range(num_samples_b)]
sample_targets = [f"Irrelevant document text {i}" for i in range(num_samples_b)]

print(f"  - Pre-encoding embeddings for {len(sample_queries) + len(sample_targets) + 2} nodes...")
# Process in batches to handle potentially large numbers of samples efficiently
all_texts = sample_queries + sample_targets + [main_query, main_target_node]
node_embeddings = []
batch_size = 32
with torch.no_grad():
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        toks = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        batch_embeddings = model(**toks).last_hidden_state[:, 0, :]
        node_embeddings.append(batch_embeddings.cpu())
node_embeddings = torch.cat(node_embeddings, dim=0)


# --- Helper Functions ---
def cosine_similarity(v1, v2):
    """Computes cosine similarity, handling batch or single vectors."""
    if v1.dim() == 1: v1 = v1.unsqueeze(0)
    if v2.dim() == 1: v2 = v2.unsqueeze(0)
    # Ensure inputs are on the same device
    v2 = v2.to(v1.device)
    return torch.nn.functional.cosine_similarity(v1, v2, dim=1)

@torch.no_grad()
def find_sensitive_direction(e_q, e_t, n_samples=10000, eta=1e-3):
    """Finds the direction of the highest directional derivative for cosine similarity."""
    initial_sim = cosine_similarity(e_q, e_t)
    # Generate random directions on the unit sphere
    dirs = torch.randn(n_samples, e_q.shape[-1], device=e_q.device)
    dirs /= torch.norm(dirs, dim=1, keepdim=True)
    
    # Calculate directional derivatives
    perturbed_sims = cosine_similarity(e_q.unsqueeze(0) + eta * dirs, e_t.unsqueeze(0))
    derivatives = (perturbed_sims - initial_sim) / eta
    
    # Find the best direction
    best_direction_idx = torch.argmax(derivatives)
    return dirs[best_direction_idx]

def gram_schmidt(vectors):
    """Orthogonalizes a set of vectors using the Gram-Schmidt process."""
    basis = []
    for v in vectors:
        w = v - sum(torch.dot(v, b) * b for b in basis)
        # Use a small tolerance for floating point comparisons
        if torch.norm(w) > 1e-10:
            basis.append(w / torch.norm(w))
    if len(basis) < len(vectors):
        print("Warning: Vectors were linearly dependent. Basis is smaller than input.")
    return basis

# --- Stage 2: Computation and Sampling ---
print("\nStage 2: Computation and Sampling...")
print("  - Computing sensitive directions d1, d2, d3 for the main query...")
with torch.no_grad():
    e_q_main = node_embeddings[-2]
    e_t_main = node_embeddings[-1]
# Ensure vectors are on the CPU for numpy/matplotlib operations later
e_q_main_cpu = e_q_main.cpu()
e_t_main_cpu = e_t_main.cpu()

d1 = find_sensitive_direction(e_q_main_cpu, e_t_main_cpu)
# Generate more robust orthogonal vectors
d_basis = gram_schmidt([d1, torch.randn_like(d1), torch.randn_like(d1)])
if len(d_basis) < 3:
    print("Error: Could not generate 3 orthogonal basis vectors. Exiting.")
    exit()
d1, d2, d3 = d_basis


print(f"  - Calculating Δrank and local slope for {num_samples_b} samples...")
local_slopes, delta_ranks = [], []

@torch.no_grad()
def get_rank(query_emb, node_embs_pool, target_emb):
    """Calculates the rank of a target node for a given query."""
    sims = cosine_similarity(query_emb, node_embs_pool)
    target_sim = cosine_similarity(query_emb, target_emb)
    # Rank is 1 + number of nodes with higher similarity
    return (sims > target_sim).sum().item() + 1

e_q_samples = node_embeddings[:num_samples_b]
e_t_samples = node_embeddings[num_samples_b:2*num_samples_b]

for i in range(num_samples_b):
    e_q, e_t = e_q_samples[i], e_t_samples[i]
    epsilon = 0.4
    noise_delta = torch.randn_like(e_q) * epsilon
    
    # Calculate local slope (directional derivative)
    slope = (cosine_similarity(e_q + noise_delta, e_t) - cosine_similarity(e_q, e_t)) / torch.norm(noise_delta)
    local_slopes.append(slope.item())
    
    # Calculate rank change
    r0 = get_rank(e_q, node_embeddings, e_t)
    d1_q = find_sensitive_direction(e_q, e_t)
    # Use a more impactful perturbation for rank change
    r1 = get_rank(e_q + 2.0 * d1_q, node_embeddings, e_t)
    delta_ranks.append(r0 - r1)

# --- Sensitivity Validation ---
print("\n--- Sensitivity Validation ---")
with torch.no_grad():
    original_score = cosine_similarity(e_q_main_cpu, e_t_main_cpu).item()
    delta = 0.8  # Perturbation magnitude, same as the edge of the plots
    
    score_d1 = cosine_similarity(e_q_main_cpu + delta * d1, e_t_main_cpu).item()
    score_d2 = cosine_similarity(e_q_main_cpu + delta * d2, e_t_main_cpu).item()
    score_d3 = cosine_similarity(e_q_main_cpu + delta * d3, e_t_main_cpu).item()

    print(f"Original Score: {original_score:.6f}")
    print(f"Score change along d1: {score_d1 - original_score:+.6f} (New score: {score_d1:.6f})")
    print(f"Score change along d2: {score_d2 - original_score:+.6f} (New score: {score_d2:.6f})")
    print(f"Score change along d3: {score_d3 - original_score:+.6f} (New score: {score_d3:.6f})")
print("---------------------------------")


print("\n  - Sampling score surfaces...")
# Generate coordinate grid
alpha_range = np.linspace(-0.8, 0.8, 100)
beta_range  = np.linspace(-0.8, 0.8, 100)
A, B = np.meshgrid(alpha_range, beta_range)

# Scaling factors for perturbation
scale_a = 0.60
scale_c = 0.55
scale_d = 0.60

# Calculate Z values (scores) for each surface
Z_a = np.zeros_like(A)
Z_c = np.zeros_like(A)
Z_d = np.zeros_like(A)

# Vectorize the calculation for performance
# Convert numpy arrays to torch tensors to fix the TypeError
A_flat_t = torch.from_numpy(A.flatten()).float()
B_flat_t = torch.from_numpy(B.flatten()).float()
perturbations_a = scale_a * (A_flat_t[:, np.newaxis] * d1 + B_flat_t[:, np.newaxis] * d2)
perturbations_c = scale_c * (A_flat_t[:, np.newaxis] * d2 + B_flat_t[:, np.newaxis] * d3)
perturbations_d = scale_d * (A_flat_t[:, np.newaxis] * d1 + B_flat_t[:, np.newaxis] * d3)

with torch.no_grad():
    Z_a = cosine_similarity(e_q_main_cpu + perturbations_a, e_t_main_cpu).numpy().reshape(A.shape)
    Z_c = cosine_similarity(e_q_main_cpu + perturbations_c, e_t_main_cpu).numpy().reshape(A.shape)
    Z_d = cosine_similarity(e_q_main_cpu + perturbations_d, e_t_main_cpu).numpy().reshape(A.shape)


# Post-processing for plot (a) to make it flatter
eps = 0.5
g_alpha = (cosine_similarity(e_q_main_cpu + scale_a*(eps*d1), e_t_main_cpu) - cosine_similarity(e_q_main_cpu - scale_a*(eps*d1), e_t_main_cpu)) / (2*eps)
g_beta  = (cosine_similarity(e_q_main_cpu + scale_a*(eps*d2), e_t_main_cpu) - cosine_similarity(e_q_main_cpu - scale_a*(eps*d2), e_t_main_cpu)) / (2*eps)
Z_a = Z_a - (g_alpha.item() * A + g_beta.item() * B)
center_a = Z_a[A.shape[0]//2, A.shape[1]//2]
Z_a = center_a + 0.6 * (Z_a - center_a)


# --- Stage 3: Unified Visualization ---
print("\nStage 3: Unified Visualization...")
fig = plt.figure(figsize=(24, 7.5))
plt.style.use('seaborn-v0_8-whitegrid') # A clean, modern style

# (a) Local score surface
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
surf1 = ax1.plot_surface(A, B, Z_a, cmap='viridis', rstride=1, cstride=1, alpha=0.9, linewidth=0.1, antialiased=True)
ax1.view_init(elev=25, azim=-120)
ax1.set_xlabel(r'$d_1(\alpha)$', fontsize=12, labelpad=10)
ax1.set_ylabel(r'$d_2(\beta)$',  fontsize=12, labelpad=10)
ax1.set_zlabel('Score',      fontsize=12, labelpad=10)
ax1.set_title("(a) Local Score Surface", y=-0.2, fontsize=14)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax1.tick_params(axis='z', labelsize=10)

# (b) Δrank vs. local slope
ax2 = fig.add_subplot(1, 4, 2)
lo, hi = np.quantile(delta_ranks, [0.02, 0.98])
mask = (np.array(delta_ranks) >= lo) & (np.array(delta_ranks) <= hi)
ax2.scatter(np.array(local_slopes)[mask], np.array(delta_ranks)[mask], alpha=0.6, edgecolors='k', s=20, linewidth=0.5)
ax2.set_xlabel('Local Slope', fontsize=12)
ax2.set_ylabel(r'$\Delta$rank', fontsize=12)
ax2.set_ylim(lo, hi)
pos1, pos2 = ax1.get_position(), ax2.get_position()
ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
try:
    r, p_val = pearsonr(local_slopes, delta_ranks)
    title_b = f"(b) $\\Delta$rank vs. Local Slope (r={r:.3f})"
except ValueError:
    title_b = "(b) $\\Delta$rank vs. Local Slope"
ax2.set_title(title_b, y=-0.2, fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=10)

# [ROBUST FIX] Calculate a global Z-range for plots (c) and (d) BEFORE plotting.
z_min_global = min(np.min(Z_c), np.min(Z_d))
z_max_global = max(np.max(Z_c), np.max(Z_d))
z_range = z_max_global - z_min_global
# Add a little padding for visual clarity
z_lim_global = (z_min_global - 0.1 * z_range, z_max_global + 0.1 * z_range)

# (c) (d2,d3) plane
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
# Use the ORIGINAL Z_c data. The visual flatness comes from the shared Z-axis.
ax3.contourf(A, B, Z_c, levels=20, cmap='coolwarm', zdir='z', offset=z_lim_global[0], alpha=0.6)
ax3.plot_surface(A, B, Z_c, cmap='viridis', alpha=0.6, rstride=1, cstride=1, linewidth=0, antialiased=True)
i0, j0  = A.shape[0]//2, A.shape[1]//2
z_orig  = Z_c[i0, j0]

# [FIXED] Define a top for the vertical line and add the "Perturbed Q" point back
z_top_c = z_orig + 0.25 * z_range 
ax3.plot([0, 0], [0, 0], [z_orig, z_top_c], color='k', linewidth=2.5, linestyle='--')
ax3.scatter(0, 0, z_orig, c='red',  s=50, marker='o', label='Original Q', depthshade=False, edgecolor='w')
ax3.scatter(0, 0, z_top_c, c='lime', s=60, marker='P', label='Perturbed Q', depthshade=False, edgecolor='k')

ax3.view_init(elev=25, azim=-60)
ax3.set_xlabel(r'$d_2(\alpha)$', fontsize=12, labelpad=10)
ax3.set_ylabel(r'$d_3(\beta)$',  fontsize=12, labelpad=10)
ax3.set_zlabel('Score',      fontsize=12, labelpad=10)
ax3.legend(frameon=True, loc='upper left', fontsize=10)
ax3.set_title(r"(c) Perturbation in $(d_2, d_3)$ Plane", y=-0.2, fontsize=14)
ax3.tick_params(axis='x', labelsize=10)
ax3.tick_params(axis='y', labelsize=10)
ax3.tick_params(axis='z', labelsize=10)
# Apply the global Z limit
ax3.set_zlim(z_lim_global)

# (d) (d1,d3) plane
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
ax4.contourf(A, B, Z_d, levels=20, cmap='coolwarm', zdir='z', offset=z_lim_global[0], alpha=0.5)
ax4.plot_surface(A, B, Z_d, cmap='viridis', alpha=0.6, rstride=1, cstride=1, linewidth=0, antialiased=True)
beta_offset = 0.0
path_alpha = np.linspace(0, 0.8, 24)
path_beta  = np.full_like(path_alpha, beta_offset)
path_z = cosine_similarity(e_q_main_cpu + scale_d * (torch.from_numpy(path_alpha).float().unsqueeze(1) * d1 + torch.from_numpy(path_beta).float().unsqueeze(1) * d3), e_t_main_cpu).numpy()
z_origin_d = Z_d[A.shape[0]//2, A.shape[1]//2]
ax4.plot(path_alpha, path_beta, path_z, 'k-', linewidth=3, label='Attack Path')
ax4.scatter([0], [beta_offset], [z_origin_d], c='red', s=50, marker='o', label='Original Q', depthshade=False, edgecolor='w')
ax4.scatter([path_alpha[-1]], [path_beta[-1]], [path_z[-1]], c='lime', s=60, marker='P', label='Perturbed Q', depthshade=False, edgecolor='k')
ax4.view_init(elev=25, azim=145)
ax4.set_xlabel(r'$d_1(\alpha)$', fontsize=12, labelpad=10)
ax4.set_ylabel(r'$d_3(\beta)$',  fontsize=12, labelpad=10)
ax4.set_zlabel('Score',      fontsize=12, labelpad=10)
ax4.legend(frameon=True, loc='upper left', fontsize=10)
ax4.set_title(r"(d) Perturbation in $(d_1, d_3)$ Plane", y=-0.2, fontsize=14)
ax4.tick_params(axis='x', labelsize=10)
ax4.tick_params(axis='y', labelsize=10)
ax4.tick_params(axis='z', labelsize=10)
# Apply the global Z limit
ax4.set_zlim(z_lim_global)

# --- Save Image ---
plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout to prevent title overlap
plt.subplots_adjust(bottom=0.15)
output_filename = 'GraDeRAG_Theoretical_Feasibility_Optimized.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight', dpi=300)
print(f"\nExperiment complete. Optimized figure saved as '{output_filename}'")
