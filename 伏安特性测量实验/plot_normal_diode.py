"""图1  普通二极管的正向伏安特性曲线。

数据：实验原始记录，保存在同目录的 data_normal_diode.csv（U 单位 V，I 单位 mA）。

方法：
1. 深蓝色散点标出实验数据点，浅蓝色实线平滑连接；
2. 取电流随电压剧烈变化的线性段（U >= 0.680 V）做最小二乘拟合 I = a*U + b，
   用红色虚线画出拟合线段（并以较淡的红色虚线延伸至 I = 0）；
3. 由拟合直线的横轴截距得正向导通阈值电压 U_th1，用黑色箭头 + 标注框标出，
   框内同时写明拟合方程；
4. 预留非理想电表负载效应修正：填入表内阻后自动叠加修正曲线（默认关闭）。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# 设置中文支持与数学符号排版
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# 以脚本所在目录为基准，与当前工作目录无关
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "data_normal_diode.csv"
RESULT_PATH = SCRIPT_DIR / "result_normal_diode.txt"

# 直线段拟合窗口（V）。下限取电流随电压剧烈变化的起点，上限 None 表示不设限。
FIT_U_MIN = 0.680
FIT_U_MAX = None
# 备用对比窗口：严格取 0.680~0.700 V 这段斜率最平直的区间
FIT_U_MIN_ALT = 0.680
FIT_U_MAX_ALT = 0.700

# 非理想电表内阻（Ω）。填入数值即可开启负载效应修正，None 表示不修正。
AMMETER_INTERNAL_R = None  # 电流表内阻 R_A
VOLTMETER_INTERNAL_R = None  # 电压表内阻 R_V
# 电压表接法：True = 电压表并在被测二极管两端；False = 电压表并在电源两端
VOLTMETER_ACROSS_DIODE = True


def read_data(path):
    """读取 U, I 两列原始数据。"""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data["U"]), np.atleast_1d(data["I"])


def apply_load_effect(u_meas, i_meas, r_a, r_v, voltmeter_across_diode):
    """修正非理想电表带来的负载效应，返回二极管真实的 (U_D, I_D)。

    电流表内阻 R_A 引起分压，电压表内阻 R_V 引起分流：
      · 电压表并在二极管两端：二极管电压读数准确，电流表多计了流过电压表的电流
            I_D = I_表 - U_D / R_V
      · 电压表并在电源两端：电流表读数准确，二极管电压需扣除电流表上的压降
            U_D = U_表 - I_表 * R_A
    """
    if voltmeter_across_diode:
        u_d = u_meas
        i_d = i_meas - u_meas / r_v if r_v else i_meas
    else:
        i_d = i_meas
        u_d = u_meas - i_meas * r_a if r_a else u_meas
    return u_d, i_d


def fit_line(u, i, u_min, u_max):
    """在给定窗口内做最小二乘拟合 I = a*U + b，返回 (a, b, U_th, R², 点数)。"""
    mask = u >= u_min
    if u_max is not None:
        mask &= u <= u_max
    a, b = np.polyfit(u[mask], i[mask], 1)
    pred = a * u[mask] + b
    ss_res = np.sum((i[mask] - pred) ** 2)
    ss_tot = np.sum((i[mask] - i[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, -b / a, r2, int(mask.sum())


def main():
    u, i = read_data(CSV_PATH)
    a, b, u_th, r2, n_fit = fit_line(u, i, FIT_U_MIN, FIT_U_MAX)
    a_alt, b_alt, u_th_alt, r2_alt, n_alt = fit_line(u, i, FIT_U_MIN_ALT, FIT_U_MAX_ALT)

    print("--- 普通二极管伏安特性拟合结果 ---")
    print(f"拟合窗口 U >= {FIT_U_MIN:.3f} V（{n_fit} 个点）")
    print(f"拟合直线方程: I = {a:.2f} U {b:+.2f}   (mA, V)")
    print(f"正向导通阈值电压 U_th1 = {u_th:.3f} V")
    print(f"决定系数 R² = {r2:.5f}")
    print(f"参考：窗口 {FIT_U_MIN_ALT:.3f}~{FIT_U_MAX_ALT:.3f} V（{n_alt} 个点）"
          f" -> U_th = {u_th_alt:.3f} V, R² = {r2_alt:.5f}")

    # ==================== 图1：正向伏安特性曲线 ====================
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # 平滑曲线：PCHIP 单调插值，既光滑又不会在拐点附近过冲到负值
    u_smooth = np.linspace(u.min(), u.max(), 600)
    i_smooth = PchipInterpolator(u, i)(u_smooth)
    ax.plot(u_smooth, i_smooth, color="deepskyblue", lw=1.6, zorder=2, label="平滑拟合曲线")

    # 实验数据点（深蓝色散点）
    ax.scatter(u, i, color="darkblue", s=42, zorder=5, label="实验测量数据点")

    # 红色虚线：直线段拟合线（实线段在数据范围内，虚线段为外推到 I = 0）
    u_fit_line = np.linspace(FIT_U_MIN, u.max(), 100)
    ax.plot(
        u_fit_line,
        a * u_fit_line + b,
        color="red",
        lw=2.0,
        ls="--",
        zorder=4,
        label=f"最小二乘拟合线 ($U\\geq{FIT_U_MIN:.3f}$ V)",
    )
    u_extrap = np.linspace(u_th, FIT_U_MIN, 50)
    ax.plot(
        u_extrap,
        a * u_extrap + b,
        color="red",
        lw=1.3,
        ls="--",
        alpha=0.6,
        zorder=4,
        label="外推至 $I=0$",
    )

    # 阈值电压：黑色箭头 + 标注框，框内写明 U_th1 与拟合方程
    ax.scatter([u_th], [0], color="black", s=55, zorder=6, marker="o")
    ax.annotate(
        f"$U_{{th1}}={u_th:.3f}\\,\\mathrm{{V}}$\n"
        f"拟合直线 $I={a:.1f}U{b:+.1f}$",
        xy=(u_th, 0),
        xytext=(0.47, 1.7),
        textcoords="data",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "black", "alpha": 0.93},
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.3, "shrinkB": 4},
        zorder=7,
    )

    # 负载效应修正曲线（填入表内阻后才画）
    if AMMETER_INTERNAL_R or VOLTMETER_INTERNAL_R:
        r_a = AMMETER_INTERNAL_R or 0.0
        r_v = VOLTMETER_INTERNAL_R or np.inf
        u_corr, i_corr = apply_load_effect(u, i, r_a, r_v, VOLTMETER_ACROSS_DIODE)
        ax.plot(u_corr, i_corr, color="tab:green", lw=1.6, ls="-.", zorder=3,
                label="非理想电表负载效应修正后")
        print(f"已叠加负载效应修正曲线 (R_A={r_a} Ω, R_V={r_v} Ω)")

    ax.set_title("图1  普通二极管的正向伏安特性曲线", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax.set_xlim(-0.02, 0.78)
    ax.set_ylim(-1.0, 15)
    ax.set_xticks(np.arange(0.0, 0.81, 0.1))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()

    out = SCRIPT_DIR / "normal_diode_iv.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out.name}")

    # ==================== 结果写入文本 ====================
    lines = [
        "普通二极管正向伏安特性\n\n",
        f"拟合窗口 U >= {FIT_U_MIN:.3f} V（{n_fit} 个点）\n",
        f"拟合直线 I = {a:.4f} * U {b:+.4f}  (mA, V)\n",
        f"U_th1 = {u_th:.4f} V\n",
        f"R2 = {r2:.6f}\n\n",
        f"参考窗口 {FIT_U_MIN_ALT:.3f}~{FIT_U_MAX_ALT:.3f} V（{n_alt} 个点）\n",
        f"I = {a_alt:.4f} * U {b_alt:+.4f}\n",
        f"U_th = {u_th_alt:.4f} V\n",
        f"R2 = {r2_alt:.6f}\n",
    ]
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    plt.show()


if __name__ == "__main__":
    main()
