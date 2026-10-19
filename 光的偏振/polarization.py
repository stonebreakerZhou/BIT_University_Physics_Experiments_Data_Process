from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 设置中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
# 数学符号（cos²φ、R² 等）交给 mathtext 排版，避免 SimHei 里 ² 字形上浮
plt.rcParams["mathtext.fontset"] = "stix"

# 以脚本所在目录为基准，与当前工作目录无关
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "malus_data.csv"

# 从 CSV 文件中读取数据（跳过表头）
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
phi_all = data[:, 0]
P_all = data[:, 1]
cos2_phi = data[:, 2]

# 1. 绘制 P - φ 关系曲线 (0° ~ 350°)
plt.figure(figsize=(9, 5))
plt.scatter(phi_all, P_all, color="black", s=30, zorder=5, label="实验数据点")

phi_smooth = np.linspace(phi_all.min(), phi_all.max(), 500)
spl = make_interp_spline(phi_all, P_all, k=3)
P_smooth = np.clip(spl(phi_smooth), 0, None)  # 功率非负，削掉样条在谷底的微小下冲
plt.plot(phi_smooth, P_smooth, color="red", linewidth=1.5, label="B 样条平滑拟合曲线")

plt.title("$P \\sim \\varphi$", fontsize=12, pad=10)
plt.xlabel("$\\varphi$", fontsize=11)
plt.ylabel("$P$", fontsize=11)
plt.xlim(0, 360)
plt.ylim(0, 1200)
plt.xticks(np.arange(0, 361, 50))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "P_phi_curve.png", dpi=300)
plt.show()

# 2. 绘制 P - cos^2(phi) 关系曲线 (0° ~ 90°)
idx_90 = 10  # 对应 0 到 90 度的范围
phi_sub = phi_all[:idx_90]
P_sub = P_all[:idx_90]
cos2_sub = cos2_phi[:idx_90]

plt.figure(figsize=(7, 5))
plt.scatter(cos2_sub, P_sub, color="black", s=40, zorder=5, label="实验数据点")

# 线性拟合
k, b = np.polyfit(cos2_sub, P_sub, 1)
x_line = np.linspace(0, 1, 100)

# 决定系数 R²
pred = k * cos2_sub + b
r2 = 1 - np.sum((P_sub - pred) ** 2) / np.sum((P_sub - P_sub.mean()) ** 2)

plt.plot(
    x_line,
    k * x_line + b,
    color="red",
    linewidth=1.5,
    label=f"线性拟合线: $y={k:.1f}x{b:+.1f}$, $R^2={r2:.4f}$",
)

plt.title("$P \\sim \\cos^{2}\\varphi$", fontsize=12, pad=10)
plt.xlabel("$\\cos^{2}\\varphi$", fontsize=11)
plt.ylabel("$P$", fontsize=11)
plt.xlim(0, 1.0)
plt.ylim(0, 1200)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "P_cos2phi_fit.png", dpi=300)
plt.show()
