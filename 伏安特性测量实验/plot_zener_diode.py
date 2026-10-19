"""2.1V 稳压二极管正、反向伏安特性曲线。

数据：实验原始记录，保存在同目录的 data_zener_diode.csv（U 单位 V，I 单位 mA，
反向电流记为负值）。

方法：
1. 橙色散点 + 深红色平滑曲线（PCHIP 单调三次插值，保证曲线不过冲），
   完整展现反向击穿区（左侧陡直稳压段）与正向导通区（右侧上升段）；
2. 十字坐标轴（x=0、y=0 加粗虚线）；
3. 稳压电压 U_Z：按稳压管数据手册惯例，取反向击穿区工作电流 Iz = 5 mA 处的
   管电压（另打印 Iz = 10 mA 的值，便于与实验指导书对照）；
4. 正向导通电压 U_F：取正向电流 I = 1 mA 处的管电压；
5. 另出反向、正向两张放大图（两支合在一张图里都被压扁，看不清拐点）；
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
CSV_PATH = SCRIPT_DIR / "data_zener_diode.csv"
RESULT_PATH = SCRIPT_DIR / "result_zener_diode.txt"

# 稳压电压 / 导通电压的测试工作点
IZ_TEST_MA = 5.0  # 稳压电压的测试工作电流（mA），小功率稳压管常用 5 mA
IZ_TEST_ALT_MA = 10.0  # 备用测试电流（mA）
IF_TEST_MA = 1.0  # 正向导通电压的测试电流（mA）

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


def draw_common(ax, u, i, u_smooth, i_smooth):
    """主图与放大图共用的绘制部分。"""
    ax.plot(u_smooth, i_smooth, color="crimson", lw=1.6, zorder=2, label="平滑拟合曲线")
    ax.scatter(u, i, color="darkorange", s=40, zorder=5, label="实验测量数据点")
    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.axvline(0, color="black", lw=1.2, ls="--", alpha=0.75, zorder=1)
    ax.set_xlabel("电压 $U$ / V", fontsize=12)
    ax.set_ylabel("电流 $I$ / mA", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)


def mark_threshold(ax, x_pos, y_pos, text, color):
    """竖虚线 + 高亮圆点 + 数值标在坐标轴下方（不压住任何数据点）。"""
    ax.axvline(x_pos, color=color, ls=":", lw=1.2, alpha=0.6, zorder=1)
    ax.scatter([x_pos], [y_pos], color=color, s=60, zorder=6, marker="o")
    ax.text(
        x_pos,
        -0.32,
        text,
        fontsize=11,
        fontweight="bold",
        color=color,
        ha="center",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": color, "alpha": 0.9},
    )


def main():
    u, i = read_data(CSV_PATH)

    u_z = reverse_voltage_at_current(u, i, IZ_TEST_MA)
    u_z_alt = reverse_voltage_at_current(u, i, IZ_TEST_ALT_MA)
    u_f = forward_voltage_at_current(u, i, IF_TEST_MA)

    print("--- 2.1V 稳压二极管伏安特性分析 ---")
    print(f"稳压电压 U_Z (Iz = {IZ_TEST_MA:.0f} mA) = {u_z:.4f} V")
    print(f"稳压电压 U_Z (Iz = {IZ_TEST_ALT_MA:.0f} mA) = {u_z_alt:.4f} V")
    print(f"正向导通电压 U_F (I = {IF_TEST_MA:.0f} mA) = {u_f:.4f} V")
    print(f"标称稳压值 2.1 V，实测偏差 {abs(u_z - 2.1):.4f} V "
          f"({abs(u_z - 2.1) / 2.1 * 100:.2f}%)")

    u_smooth = np.linspace(u.min(), u.max(), 900)
    i_smooth = PchipInterpolator(u, i)(u_smooth)

    # ==================== 图 1：完整伏安特性 ====================
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_common(ax, u, i, u_smooth, i_smooth)

    # 稳压电压与正向导通电压
    mark_threshold(
        ax, -u_z, -IZ_TEST_MA,
        f"$U_Z={u_z:.3f}\\,\\mathrm{{V}}$", "red",
    )
    mark_threshold(
        ax, u_f, IF_TEST_MA,
        f"$U_F={u_f:.3f}\\,\\mathrm{{V}}$", "tab:green",
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
    ax.set_xlim(-2.5, 1.0)
    ax.set_ylim(-14, 15)
    ax.set_xticks(np.arange(-2.5, 1.01, 0.5))
    ax.legend(loc="upper left", fontsize=10.5)
    fig.tight_layout()
    out_main = SCRIPT_DIR / "zener_diode_iv.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out_main.name}")

    # ==================== 图 2：反向伏安特性放大 ====================
    fig2, ax2 = plt.subplots(figsize=(8.5, 6))
    rev = u <= 0.0
    draw_common(ax2, u[rev], i[rev], u_smooth[u_smooth <= 0.0], i_smooth[u_smooth <= 0.0])

    # 反向截止区（|I| < 0.01 mA）用浅色标出
    ax2.axvspan(-1.0, 0.0, color="0.92", zorder=0)
    ax2.text(-0.52, -11.0, "反向截止区", fontsize=11, ha="center", va="center", color="0.35")

    # 稳压工作点
    mark_threshold(
        ax2, -u_z, -IZ_TEST_MA,
        f"$U_Z={u_z:.3f}\\,\\mathrm{{V}}$\n($I_z={IZ_TEST_MA:.0f}\\,\\mathrm{{mA}}$)", "red",
    )
    ax2.text(-2.2, -2.6, "反向击穿区（稳压区）", fontsize=11, ha="center", va="center", color="red")

    ax2.set_title("图2  稳压二极管反向伏安特性（放大）", fontsize=13.5, fontweight="bold", pad=12)
    ax2.set_xlim(-2.4, 0.05)
    ax2.set_ylim(-13.5, 1.2)
    ax2.legend(loc="upper right", fontsize=10.5)
    fig2.tight_layout()
    out_rev = SCRIPT_DIR / "zener_diode_iv_reverse.png"
    fig2.savefig(out_rev, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out_rev.name}")

    # ==================== 图 3：正向伏安特性放大 ====================
    fig3, ax3 = plt.subplots(figsize=(8.5, 6))
    fwd = u > 0.0
    draw_common(ax3, u[fwd], i[fwd], u_smooth[u_smooth > 0.0], i_smooth[u_smooth > 0.0])

    mark_threshold(
        ax3, u_f, IF_TEST_MA,
        f"$U_F={u_f:.3f}\\,\\mathrm{{V}}$\n($I={IF_TEST_MA:.0f}\\,\\mathrm{{mA}}$)", "tab:green",
    )

    ax3.set_title("图2  稳压二极管正向伏安特性（放大）", fontsize=13.5, fontweight="bold", pad=12)
    ax3.set_xlim(0.5, 0.82)
    ax3.set_ylim(-1.0, 14.5)
    ax3.legend(loc="upper left", fontsize=10.5)
    fig3.tight_layout()
    out_fwd = SCRIPT_DIR / "zener_diode_iv_forward.png"
    fig3.savefig(out_fwd, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存: {out_fwd.name}")

    # ==================== 结果写入文本 ====================
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("2.1V 稳压二极管正、反向伏安特性\n")
        f.write(f"稳压电压 U_Z (Iz={IZ_TEST_MA:.0f} mA) = {u_z:.4f} V\n")
        f.write(f"稳压电压 U_Z (Iz={IZ_TEST_ALT_MA:.0f} mA) = {u_z_alt:.4f} V\n")
        f.write(f"正向导通电压 U_F (I={IF_TEST_MA:.0f} mA) = {u_f:.4f} V\n")
        f.write(f"标称稳压值偏差 = {abs(u_z - 2.1):.4f} V "
                f"({abs(u_z - 2.1) / 2.1 * 100:.2f}%)\n")

    plt.show()


if __name__ == "__main__":
    main()
