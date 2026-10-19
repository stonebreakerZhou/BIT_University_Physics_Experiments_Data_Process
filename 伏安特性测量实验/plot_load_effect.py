"""图4  普通二极管密集伏安特性曲线及换档直线交点图。

复刻老师 PPT 第 35 页的版式：左侧为 0~15 V 全景图（含两条换档负载线），
右侧叠放交点区域（0.4~0.7 V）的放大子图，用蓝色箭头"放大后"连接。

物理含义：电源经限流电阻与电流表内阻串联后给二极管供电，工作点满足
    I = (VCC - U) / R      (R 为该档位下的总串联电阻，单位 kΩ)
即伏安特性曲线与"换档负载线"的交点。直线由坐标轴上两点确定：
    U 轴交点 (VCC, 0 mA)，i 轴交点 (0 V, VCC / R)
老师 PPT 要求：直线作图横轴坐标必须到 15 V。

数据：实验原始记录，保存在同目录的 data_normal_diode_dense.csv（U 单位 V，I 单位 mA）。

输出：
    load_effect_intersection.png   图4 全景 + 放大子图
    result_load_effect.txt         交点核验结果
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
CSV_PATH = SCRIPT_DIR / "data_normal_diode_dense.csv"
RESULT_PATH = SCRIPT_DIR / "result_load_effect.txt"

# 两条换档负载线：(VCC, 总串联电阻 kΩ, 颜色, 图例文字)
LOAD_LINES = [
    (15.00, 6.663, "red", "$15.00\\,\\mathrm{V}$ 换档后直线"),
    (4.44, 1.108, "green", "$4.44\\,\\mathrm{V}$ 换档后直线"),
]

# 三个关键工作点：(U, I, 颜色, 标注文字, 子图内文本框位置)
# 工作点0 = 换档前的工作点，15 V 与 4.44 V 两种电压必须用同一个点（老师 PPT 要求）。
# u1 = 0.623 V 与 i1 = 1.901 mA 互为自洽：R = (15.00-0.623)/1.901 = 7563 Ω，
# R-900 = 6663 Ω；4.44 V 时 R = (4.44-0.623)/1.901 = 2008 Ω，R-900 = 1108 Ω。
WORK_POINTS = [
    (0.623, 1.901, "black", "工作点0\n1.901mA\n0.623V", (0.578, 1.45)),
    (0.630, 2.13, "red", "15.00V\n换档后\n2.13mA\n0.630V", (0.585, 2.60)),
    (0.655, 3.33, "green", "4.44V\n换档后\n3.33mA\n0.655V", (0.615, 3.62)),
]

# 全景图 / 放大子图的显示区间
X_LIM_MAIN = (0.0, 15.2)
Y_LIM_MAIN = (0.0, 15.5)
X_LIM_INSET = (0.40, 0.70)
Y_LIM_INSET = (0.0, 4.0)

# 放大子图在主图坐标系中的位置（主图 axes 比例）
INSET_BOX = [0.44, 0.40, 0.53, 0.55]

# 全景图上标出"待放大"区域的方框（主图数据坐标）
ZOOM_MARK = (0.30, 0.90, 0.0, 4.3)


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


def load_line_current(vcc, resistance, u):
    """负载线电流 I = (VCC - U) / R，单位 mA。"""
    return (vcc - u) / resistance


def find_intersections(i_curve, vcc, resistance, u_grid):
    """求负载线与伏安特性曲线的全部交点，返回 [(U, I), ...]。

    在 u_grid 上找负载线电流与曲线电流的符号变化点，再用二分法细化。
    """
    def f(u):
        return float(load_line_current(vcc, resistance, u)) - float(i_curve(u))

    hits = []
    for k in range(len(u_grid) - 1):
        f0, f1 = f(u_grid[k]), f(u_grid[k + 1])
        if f0 == 0.0:
            hits.append((float(u_grid[k]), float(load_line_current(vcc, resistance, u_grid[k]))))
        elif f0 * f1 < 0.0:
            lo, hi = float(u_grid[k]), float(u_grid[k + 1])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(lo) * f(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            u_star = 0.5 * (lo + hi)
            hits.append((u_star, float(load_line_current(vcc, resistance, u_star))))
    return hits


def draw_main_panel(ax, u, i, u_smooth, i_smooth):
    """左侧全景图：0~15 V，横轴必须延伸到 15 V（老师 PPT 明确要求）。"""
    # 密集伏安特性曲线（全景下呈竖直尖峰）
    ax.plot(u_smooth, i_smooth, color="black", lw=1.2, zorder=3)
    ax.scatter(u, i, color="black", s=22, zorder=4)

    # 两条换档负载线：由坐标轴两点确定，横轴延伸到 15 V
    for vcc, resistance, color, _label in LOAD_LINES:
        u_line = np.linspace(0.0, vcc, 400)
        ax.plot(
            u_line,
            load_line_current(vcc, resistance, u_line),
            color=color,
            lw=1.8,
            zorder=5,
        )

    # 直线方程的文字标注（分别放在各自直线的空白侧）
    ax.text(
        6.2, 1.75, "15.00V换档后直线方程", color="red", fontsize=12, fontweight="bold",
        ha="center", va="center",
    )
    ax.text(
        1.6, 0.45, "4.44V换档后直线方程", color="green", fontsize=12, fontweight="bold",
        ha="center", va="center",
    )

    # 标出待放大的区域
    ax.add_patch(
        plt.Rectangle(
            (ZOOM_MARK[0], ZOOM_MARK[1]),
            ZOOM_MARK[2] - ZOOM_MARK[0],
            ZOOM_MARK[3] - ZOOM_MARK[1],
            fill=False,
            edgecolor="black",
            lw=1.4,
            zorder=6,
        )
    )

    ax.set_xlim(*X_LIM_MAIN)
    ax.set_ylim(*Y_LIM_MAIN)
    ax.set_xticks(np.arange(0, 16, 1))
    ax.set_yticks(np.arange(0, 16, 1))
    ax.set_xlabel("电压 $U$ / V", fontsize=13)
    ax.set_ylabel("电流 $I$ / mA", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)


def draw_inset_panel(ax, u, i, u_smooth, i_smooth):
    """右侧放大子图：0.4~0.7 V，标出三个工作点及其水平/竖直虚线。"""
    ax.plot(u_smooth, i_smooth, color="black", lw=1.4, zorder=3)
    ax.scatter(u, i, color="black", s=28, zorder=4)

    for u_pt, i_pt, color, text, (tx, ty) in WORK_POINTS:
        # 水平虚线：从坐标轴左边缘拉到工作点
        ax.plot(
            [X_LIM_INSET[0], u_pt],
            [i_pt, i_pt],
            color=color,
            ls="--",
            lw=1.3,
            alpha=0.9,
            zorder=5,
        )
        # 竖直虚线：从工作点垂直落到横轴
        ax.plot(
            [u_pt, u_pt],
            [0.0, i_pt],
            color=color,
            ls="--",
            lw=1.3,
            alpha=0.9,
            zorder=5,
        )
        # 工作点圆圈
        ax.plot(
            [u_pt], [i_pt],
            "o",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.8,
            markersize=11,
            zorder=6,
        )
        # 标注框
        ax.annotate(
            text,
            xy=(u_pt, i_pt),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": color, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "shrinkB": 4},
            zorder=7,
        )

    ax.text(
        0.695,
        0.12,
        "换档后的计算值 $(u',\\ i')$\n图解法求交点",
        fontsize=9,
        ha="right",
        va="bottom",
        color="0.25",
        linespacing=1.6,
    )

    ax.set_xlim(*X_LIM_INSET)
    ax.set_ylim(*Y_LIM_INSET)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_yticks(np.arange(0.0, 4.01, 0.2))
    ax.set_xlabel("电压 $U$ / V", fontsize=11)
    ax.set_ylabel("电流 $I$ / mA", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(labelsize=8.5)


def main():
    u, i = read_data(CSV_PATH)

    # PCHIP 单调插值：密集数据在拐点处用普通样条会振荡
    i_curve = PchipInterpolator(u, i)
    u_smooth = np.linspace(u.min(), u.max(), 2000)
    i_smooth = i_curve(u_smooth)

    # ==================== 交点核验 ====================
    u_grid = np.linspace(u.min(), u.max(), 2000)
    intersections = {}
    for vcc, resistance, *_ in LOAD_LINES:
        hits = find_intersections(i_curve, vcc, resistance, u_grid)
        intersections[vcc] = hits
        desc = ", ".join(f"(U={uu:.4f} V, I={ii:.3f} mA)" for uu, ii in hits) or "无交点"
        print(f"{vcc:>6.2f} V 负载线与伏安特性曲线的交点: {desc}")
        print(f"        i 轴交点 (0 V, {vcc / resistance:.6f} mA)")

    print("\n--- 三个标注工作点与负载线的一致性核验 ---")
    warnings = []
    for u_pt, i_pt, _color, text, *_ in WORK_POINTS:
        curve_val = float(i_curve(u_pt))
        row = [
            f"{text.splitlines()[0]:<16s} 标注 ({u_pt:.3f} V, {i_pt:.3f} mA) | "
            f"曲线值 {curve_val:.3f} mA"
        ]
        for vcc, resistance, *_ in LOAD_LINES:
            line_val = float(load_line_current(vcc, resistance, np.array([u_pt]))[0])
            row.append(f"{vcc:.2f}V 线 {line_val:.3f} mA")
            if abs(line_val - i_pt) > 0.05:
                warnings.append(
                    f"  ! {text.splitlines()[0]}: 标注 I={i_pt:.3f} mA 与 "
                    f"{vcc:.2f}V 负载线在该处的 {line_val:.3f} mA 相差 "
                    f"{abs(line_val - i_pt):.3f} mA"
                )
        print(" | ".join(row))
    if warnings:
        print("\n提示（标注值与负载线存在偏差，请确认取点方式）：")
        for w in warnings:
            print(w)

    # ==================== 图 4：全景 + 放大子图 ====================
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.055, 0.11, 0.90, 0.78])
    draw_main_panel(ax, u, i, u_smooth, i_smooth)

    # 放大子图叠放在主图右上方（与老师 PPT 版式一致）
    ax_inset = ax.inset_axes(INSET_BOX)
    draw_inset_panel(ax_inset, u, i, u_smooth, i_smooth)
    ax_inset.set_facecolor("white")

    # 蓝色大箭头：从待放大区域指向放大子图，标注"放大后"
    arrow_to = (
        INSET_BOX[0] * (X_LIM_MAIN[1] - X_LIM_MAIN[0]) + X_LIM_MAIN[0],
        INSET_BOX[1] * (Y_LIM_MAIN[1] - Y_LIM_MAIN[0]) + Y_LIM_MAIN[0],
    )
    ax.annotate(
        "",
        xy=(arrow_to[0] - 0.4, arrow_to[1] - 0.4),
        xytext=(1.45, 3.9),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "royalblue",
            "lw": 3.0,
            "mutation_scale": 32,
            "shrinkA": 6,
            "shrinkB": 6,
        },
        zorder=7,
    )
    ax.text(
        3.6, 5.0, "放大后", color="royalblue", fontsize=14, fontweight="bold",
        ha="center", va="center", zorder=7,
        bbox={"boxstyle": "square,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )

    fig.suptitle(
        "图4  普通二极管密集伏安特性曲线及换档直线交点图",
        fontsize=15,
        fontweight="bold",
        y=0.965,
    )

    out = SCRIPT_DIR / "load_effect_intersection.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n已保存: {out.name}")

    # ==================== 结果写入文本 ====================
    lines = ["普通二极管密集伏安特性及换档直线交点\n", "\n", "换档负载线：\n"]
    for vcc, resistance, *_ in LOAD_LINES:
        lines.append(
            f"  VCC = {vcc:.2f} V, R = {resistance:.3f} kΩ"
            f"  ->  I = ({vcc:.2f} - U) / {resistance:.3f}\n"
            f"    U 轴交点 ({vcc:.2f} V, 0 mA), i 轴交点 (0 V, {vcc / resistance:.6f} mA)\n"
        )
    lines.append("\n负载线与伏安特性曲线的交点：\n")
    for vcc, hits in intersections.items():
        for uu, ii in hits:
            lines.append(f"  {vcc:.2f}V 线: U = {uu:.4f} V, I = {ii:.4f} mA\n")
    lines.append("\n关键工作点：\n")
    for u_pt, i_pt, _color, text, *_ in WORK_POINTS:
        lines.append(f"  {text.splitlines()[0]}: U = {u_pt:.3f} V, I = {i_pt:.3f} mA\n")
        lines.append(f"    该处伏安特性曲线电流 = {float(i_curve(u_pt)):.4f} mA\n")
        for vcc, resistance, *_ in LOAD_LINES:
            val = float(load_line_current(vcc, resistance, np.array([u_pt]))[0])
            lines.append(
                f"    {vcc:.2f}V 负载线电流 = {val:.4f} mA（差 {val - i_pt:+.4f} mA）\n"
            )
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    plt.show()


if __name__ == "__main__":
    main()
