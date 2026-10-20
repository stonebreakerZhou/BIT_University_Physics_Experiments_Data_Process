"""图3  红色发光二极管（LED）的正向伏安特性曲线。

数据：实验原始记录，保存在同目录的 data_led.csv（U 单位 V，I 单位 mA）。

方法：
1. 绿色散点 + 浅绿色实线平滑连接；
2. 取电流随电压剧烈变化的线性段（U >= 1.82 V）做最小二乘拟合 I = a*U + b，
   用红色虚线画出拟合线段；
3. 由拟合直线的横轴截距得正向导通阈值电压 U_th2；
4. 由 U_th2 计算光谱峰值波长 λ0 = 1240 / U_th2 (nm)，用箭头 + 标注框标出；
5. 预留非理想电表负载效应修正：填入表内阻后自动叠加修正曲线（默认关闭）。
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
CSV_PATH = SCRIPT_DIR / "data_led.csv"
RESULT_PATH = SCRIPT_DIR / "result_led.txt"

# 直线段拟合窗口（V）。下限取电流随电压剧烈变化的起点，上限 None 表示不设限。
FIT_U_MIN = 1.82
FIT_U_MAX = None

# 峰值波长计算式 λ0 = K / U_th，K 的单位为 eV·nm
WL_CONST_EV_NM = 1240.0

# 非理想电表内阻（Ω）。填入数值即可开启负载效应修正，None 表示不修正。
AMMETER_INTERNAL_R = None  # 电流表内阻 R_A
VOLTMETER_INTERNAL_R = None  # 电压表内阻 R_V
# 电压表接法：True = 电压表并在被测二极管两端；False = 电压表并在电源两端
VOLTMETER_ACROSS_DIODE = True


def read_data(path):
    """读取 U, I 两列原始数据。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件 {path.name}，请把实验记录的 (U, I) 填入该 csv（表头 U,I）。"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    u = np.atleast_1d(data["U"])
    i = np.atleast_1d(data["I"])
    if u.size == 0:
        raise ValueError(f"{path.name} 里还没有数据行，请把实验记录的 (U, I) 填进去。")
    return u, i


def apply_load_effect(u_meas, i_meas, r_a, r_v, voltmeter_across_diode):
    """修正非理想电表带来的负载效应，返回二极管真实的 (U_D, I_D)。"""
    if voltmeter_across_diode:
        u_d = u_meas
        i_d = i_meas - u_meas / r_v if r_v else i_meas
    else:
        i_d = i_meas
        u_d = u_meas - i_meas * r_a if r_a else u_meas
    return u_d, i_d


def main():
    u, i = read_data(CSV_PATH)

    # 最小二乘拟合 I = a*U + b
    mask = u >= FIT_U_MIN
    if FIT_U_MAX is not None:
        mask &= u <= FIT_U_MAX
    a, b = np.polyfit(u[mask], i[mask], 1)
    pred = a * u[mask] + b
    ss_res = np.sum((i[mask] - pred) ** 2)
    ss_tot = np.sum((i[mask] - i[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0

    u_th = -b / a  # 拟合直线在横轴上的截距 = 正向导通阈值电压
    lambda0 = WL_CONST_EV_NM / u_th  # 峰值波长 (nm)

    print("--- 红色发光二极管真实数据拟合结果 ---")
    print(f"拟合窗口 U >= {FIT_U_MIN:.3f} V（{int(mask.sum())} 个点）")
    print(f"拟合直线方程: I = {a:.2f} U {b:+.2f}   (mA, V)")
    print(f"正向导通阈值电压 U_th2 = {u_th:.3f} V")
    print(f"决定系数 R² = {r2:.5f}")
    print(f"峰值波长 λ0 = {WL_CONST_EV_NM:.0f} / {u_th:.3f} = {lambda0:.2f} nm")

    # ==================== 图3：LED 正向伏安特性曲线 ====================
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # 平滑曲线：PCHIP 单调插值，LED 拐点处极陡，样条会振荡出负电流
    u_smooth = np.linspace(u.min(), u.max(), 600)
    i_smooth = PchipInterpolator(u, i)(u_smooth)
    ax.plot(u_smooth, i_smooth, color="limegreen", lw=1.7, zorder=2, label="平滑拟合曲线")

    # 实验数据点
    ax.scatter(u, i, color="forestgreen", s=42, zorder=5, label="实验测量数据点")

    # 最小二乘拟合直线（红色虚线），从 U_th 画起以便看到外推到横轴的那一段
    u_line = np.linspace(u_th, u.max(), 120)
    ax.plot(
        u_line,
        a * u_line + b,
        color="red",
        lw=2.0,
        ls="--",
        zorder=4,
        label=f"最小二乘拟合线 ($U\\geq{FIT_U_MIN:.2f}$ V)",
    )

    # 阈值电压与峰值波长：箭头 + 标注框
    ax.scatter([u_th], [0], color="red", s=60, zorder=6, marker="o")
    ax.annotate(
        f"$U_{{th2}}={u_th:.3f}\\,\\mathrm{{V}}$\n"
        f"$\\lambda_0={WL_CONST_EV_NM:.0f}/U_{{th2}}={lambda0:.2f}\\,\\mathrm{{nm}}$",
        xy=(u_th, 0),
        xytext=(0.62, 8.6),
        textcoords="data",
        fontsize=11,
        fontweight="bold",
        color="red",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "red", "alpha": 0.93},
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.4, "shrinkB": 4},
        zorder=7,
    )

    # 负载效应修正曲线（填入表内阻后才画）
    if AMMETER_INTERNAL_R or VOLTMETER_INTERNAL_R:
        r_a = AMMETER_INTERNAL_R or 0.0
        r_v = VOLTMETER_INTERNAL_R or np.inf
        u_corr, i_corr = apply_load_effect(u, i, r_a, r_v, VOLTMETER_ACROSS_DIODE)
        ax.plot(u_corr, i_corr, color="tab:blue", lw=1.4, ls="-.", zorder=3,
                label="非理想电表负载效应修正后")
        print(f"已叠加负载效应修正曲线 (R_A={r_a} Ω, R_V={r_v} Ω)")

    ax.set_title(
        "图3  红色发光二极管（LED）的正向伏安特性曲线",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-1, 14)
    ax.set_xticks(np.arange(0.0, 2.01, 0.2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()

    out = SCRIPT_DIR / "led_iv.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out.name}")

    # ==================== 结果写入文本 ====================
    lines = [
        "红色发光二极管正向伏安特性\n\n",
        f"拟合窗口 U >= {FIT_U_MIN:.3f} V（{int(mask.sum())} 个点）\n",
        f"拟合直线 I = {a:.4f} * U {b:+.4f}  (mA, V)\n",
        f"U_th2 = {u_th:.4f} V\n",
        f"lambda_0 = {WL_CONST_EV_NM:.0f} / U_th2 = {lambda0:.2f} nm\n",
        f"R2 = {r2:.6f}\n",
    ]
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    plt.show()


if __name__ == "__main__":
    main()
