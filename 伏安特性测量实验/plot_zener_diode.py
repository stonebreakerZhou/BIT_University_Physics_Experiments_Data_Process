"""图2  2.1V 稳压二极管的正、反向伏安特性曲线。

数据：实验原始记录，保存在同目录的 data_zener_diode.csv（U 单位 V，I 单位 mA，
反向电流记为负值）。

方法：
1. 橙色散点 + 深红色（crimson）平滑曲线（PCHIP 单调三次插值，保证曲线不过冲），
   完整展现反向击穿区（左侧陡直稳压段）与正向导通区（右侧上升段）；
2. 十字坐标轴（x=0、y=0 加粗虚线）+ 背景网格；
3. 标出稳定电压 U_W ≈ -2.1 V（反向击穿区的稳压工作点）；
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
CSV_PATH = SCRIPT_DIR / "data_zener_diode.csv"
RESULT_PATH = SCRIPT_DIR / "result_zener_diode.txt"

# 稳压管标称稳压值（V），用于在图上标出稳定电压
U_W_NOMINAL = -2.1
# 稳压电压的测试工作电流（mA），小功率稳压管常用 5 mA
IZ_TEST_MA = 5.0

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
    """修正非理想电表带来的负载效应，返回二极管真实的 (U_D, I_D)。"""
    if voltmeter_across_diode:
        u_d = u_meas
        i_d = i_meas - u_meas / r_v if r_v else i_meas
    else:
        i_d = i_meas
        u_d = u_meas - i_meas * r_a if r_a else u_meas
    return u_d, i_d


def reverse_voltage_at_current(u, i, target_ma):
    """在反向支路上求 |I| = target_ma 时的稳压电压（返回正数，单位 V）。"""
    rev = u <= 0.0
    iz = -i[rev][::-1]  # 反向电流绝对值，升序
    uz = -u[rev][::-1]  # 对应的电压绝对值，升序
    return float(np.interp(target_ma, iz, uz))


def forward_voltage_at_current(u, i, target_ma):
    """在正向支路上求 I = target_ma 时的导通电压（单位 V）。"""
    fwd = u > 0.0
    return float(np.interp(target_ma, i[fwd], u[fwd]))


def main():
    u, i = read_data(CSV_PATH)

    u_z = reverse_voltage_at_current(u, i, IZ_TEST_MA)
    u_f = forward_voltage_at_current(u, i, 1.0)

    print("--- 2.1V 稳压二极管伏安特性分析 ---")
    print(f"稳压电压 U_Z (Iz = {IZ_TEST_MA:.0f} mA) = {u_z:.4f} V")
    print(f"标称稳压值 {abs(U_W_NOMINAL):.1f} V，Iz={IZ_TEST_MA:.0f} mA 实测偏差 "
          f"{abs(u_z) - abs(U_W_NOMINAL):+.4f} V "
          f"({(abs(u_z) - abs(U_W_NOMINAL)) / abs(U_W_NOMINAL) * 100:+.2f}%)")
    print(f"正向导通电压 U_F (I = 1 mA) = {u_f:.4f} V")

    u_smooth = np.linspace(u.min(), u.max(), 900)
    i_smooth = PchipInterpolator(u, i)(u_smooth)

    # ==================== 图2：正、反向伏安特性曲线 ====================
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(u_smooth, i_smooth, color="crimson", lw=1.6, zorder=2, label="平滑拟合曲线")
    ax.scatter(u, i, color="darkorange", s=42, zorder=5, label="实验测量数据点")

    # 十字坐标轴（加粗虚线）
    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.75, zorder=1)

    # 稳定电压 U_W ≈ -2.1 V：竖虚线标出位置，标出曲线上的对应工作点
    i_at_uw = float(PchipInterpolator(u, i)(U_W_NOMINAL))
    ax.axvline(U_W_NOMINAL, color="red", ls=":", lw=1.3, alpha=0.7, zorder=1)
    ax.scatter([U_W_NOMINAL], [i_at_uw], color="red", s=60, zorder=6, marker="o")
    ax.annotate(
        f"稳定电压 $U_W\\approx{U_W_NOMINAL:.1f}\\,\\mathrm{{V}}$\n"
        f"($I_z={IZ_TEST_MA:.0f}\\,\\mathrm{{mA}}$ 时 $U_Z={u_z:.3f}\\,\\mathrm{{V}}$)",
        xy=(U_W_NOMINAL, i_at_uw),
        xytext=(-1.35, -10.2),
        fontsize=10.5,
        fontweight="bold",
        color="red",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "red", "alpha": 0.93},
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.3, "shrinkB": 4},
        zorder=7,
    )

    # 正向导通电压
    ax.scatter([u_f], [1.0], color="tab:green", s=55, zorder=6, marker="o")
    ax.annotate(
        f"正向导通电压\n$U_F={u_f:.3f}\\,\\mathrm{{V}}$",
        xy=(u_f, 1.0),
        xytext=(0.62, 4.2),
        fontsize=10.5,
        fontweight="bold",
        color="tab:green",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "tab:green", "alpha": 0.93},
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 1.3, "shrinkB": 4},
        zorder=7,
    )

    # 负载效应修正曲线（填入表内阻后才画）
    if AMMETER_INTERNAL_R or VOLTMETER_INTERNAL_R:
        r_a = AMMETER_INTERNAL_R or 0.0
        r_v = VOLTMETER_INTERNAL_R or np.inf
        u_corr, i_corr = apply_load_effect(u, i, r_a, r_v, VOLTMETER_ACROSS_DIODE)
        ax.plot(u_corr, i_corr, color="tab:green", lw=1.4, ls="-.", zorder=3,
                label="非理想电表负载效应修正后")
        print(f"已叠加负载效应修正曲线 (R_A={r_a} Ω, R_V={r_v} Ω)")

    ax.set_title(
        "图2  2.1V 稳压二极管的正、反向伏安特性曲线",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax.set_xlim(-2.5, 1.0)
    ax.set_ylim(-14, 15)
    ax.set_xticks(np.arange(-2.5, 1.01, 0.5))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()

    out = SCRIPT_DIR / "zener_diode_iv.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out.name}")

    # ==================== 结果写入文本 ====================
    lines = [
        "2.1V 稳压二极管正、反向伏安特性\n\n",
        f"稳定电压 U_W ≈ {U_W_NOMINAL:.1f} V（标称）\n",
        f"稳压电压 U_Z (Iz={IZ_TEST_MA:.0f} mA) = {u_z:.4f} V\n",
        f"与标称值偏差 = {abs(u_z) - abs(U_W_NOMINAL):+.4f} V "
        f"({(abs(u_z) - abs(U_W_NOMINAL)) / abs(U_W_NOMINAL) * 100:+.2f}%)\n",
        f"正向导通电压 U_F (I=1 mA) = {u_f:.4f} V\n",
    ]
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    plt.show()


if __name__ == "__main__":
    main()
