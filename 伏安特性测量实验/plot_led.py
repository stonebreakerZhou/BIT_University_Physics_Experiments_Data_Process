"""图3  红色发光二极管的正向伏安特性曲线。

数据：实验原始记录，保存在同目录的 data_led.csv（U 单位 V，I 单位 mA）。

方法：
1. 绿色散点 + 浅绿色平滑曲线（PCHIP 单调三次插值，拐点处不过冲、不会出负电流）；
2. 取电流随电压剧烈变化的线性段（U >= 1.82 V）做最小二乘拟合 I = a*U + b；
3. 由拟合直线的横轴截距得正向导通阈值电压 U_th；
4. 由 U_th 计算光谱峰值波长 λ0 = 1240 / U_th (nm)，用箭头 + 文本框高亮标出；
5. 另出拐点局部放大图（主图里 1.6~1.87 V 只占图宽的一小段，看不清导通过程）；
6. 预留非理想电表负载效应修正：填入表内阻后自动叠加修正曲线（默认关闭）。
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


def mark_threshold(ax, x_pos, y_pos, text, color):
    """竖虚线 + 高亮圆点 + 数值标在坐标轴下方（不压住任何数据点）。"""
    ax.axvline(x_pos, color=color, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax.scatter([x_pos], [y_pos], color=color, s=60, zorder=6, marker="o")
    ax.text(
        x_pos,
        -0.25,
        text,
        fontsize=11,
        fontweight="bold",
        color=color,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": color, "alpha": 0.92},
    )


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
    print(f"正向导通阈值电压 U_th = {u_th:.3f} V")
    print(f"决定系数 R² = {r2:.5f}")
    print(f"峰值波长 λ0 = {WL_CONST_EV_NM:.0f} / {u_th:.3f} = {lambda0:.2f} nm")

    # 平滑曲线：PCHIP 单调插值，LED 拐点处极陡，样条会振荡出负电流
    u_smooth = np.linspace(u.min(), u.max(), 800)
    i_smooth = PchipInterpolator(u, i)(u_smooth)

    # ==================== 图 3：LED 正向伏安特性曲线 ====================
    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.plot(u_smooth, i_smooth, color="limegreen", lw=1.7, zorder=2, label="平滑拟合曲线")
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
        label=f"线性拟合线: $I={a:.1f}U{b:+.1f}$",
    )

    # 阈值电压与峰值波长：箭头 + 文本框，标在坐标轴下方，避开所有数据点
    ax.scatter([u_th], [0], color="red", s=60, zorder=6, marker="o")
    ax.annotate(
        f"$U_{{th}}={u_th:.3f}\\,\\mathrm{{V}}$\n"
        f"$\\lambda_0={WL_CONST_EV_NM:.0f}/U_{{th}}={lambda0:.1f}\\,\\mathrm{{nm}}$",
        xy=(u_th, 0),
        xytext=(u_th, -0.25),
        textcoords="data",
        fontsize=11,
        fontweight="bold",
        color="red",
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "red", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.4, "shrinkB": 3},
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
        "图3  红色发光二极管的正向伏安特性曲线",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-1.8, 14)
    ax.set_xticks(np.arange(0.0, 2.01, 0.2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()

    out = SCRIPT_DIR / "led_iv.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out.name}")

    # ==================== 图 3 附：导通拐点局部放大 ====================
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    ax2.plot(u_smooth, i_smooth, color="limegreen", lw=1.9, zorder=2, label="平滑拟合曲线")
    ax2.scatter(u, i, color="forestgreen", s=45, zorder=5, label="实验测量数据点")
    ax2.scatter(
        u[mask],
        i[mask],
        facecolor="none",
        edgecolor="forestgreen",
        s=150,
        lw=1.6,
        zorder=6,
        label="用于拟合的直线段数据点",
    )
    u_line2 = np.linspace(u_th, u.max(), 120)
    ax2.plot(u_line2, a * u_line2 + b, color="red", lw=2.0, ls="--", zorder=4,
             label="最小二乘拟合直线")
    ax2.scatter([u_th], [0], color="red", s=60, zorder=6, marker="o")
    ax2.annotate(
        f"$U_{{th}}={u_th:.3f}\\,\\mathrm{{V}}$\n"
        f"$\\lambda_0={lambda0:.1f}\\,\\mathrm{{nm}}$",
        xy=(u_th, 0),
        xytext=(u_th, -0.25),
        textcoords="data",
        fontsize=11,
        fontweight="bold",
        color="red",
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "red", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.4, "shrinkB": 3},
    )
    ax2.set_title(
        f"图3  红色发光二极管正向伏安特性曲线（导通拐点放大，$U\\geq{FIT_U_MIN:.2f}\\,\\mathrm{{V}}$）",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("电压 $U$ / V", fontsize=12)
    ax2.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax2.set_xlim(1.70, 1.88)
    ax2.set_ylim(-1.8, 13.5)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=10.5)
    fig2.tight_layout()
    out_zoom = SCRIPT_DIR / "led_iv_knee_zoom.png"
    fig2.savefig(out_zoom, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out_zoom.name}")

    # ==================== 结果写入文本 ====================
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("红色发光二极管正向伏安特性\n")
        f.write(f"拟合窗口 U >= {FIT_U_MIN:.3f} V（{int(mask.sum())} 个点）\n")
        f.write(f"拟合直线 I = {a:.4f} * U {b:+.4f}  (mA, V)\n")
        f.write(f"U_th = {u_th:.4f} V\n")
        f.write(f"lambda_0 = {WL_CONST_EV_NM:.0f} / U_th = {lambda0:.2f} nm\n")
        f.write(f"R2 = {r2:.6f}\n")

    plt.show()


if __name__ == "__main__":
    main()
