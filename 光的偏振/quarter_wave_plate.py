"""λ/4 波片转动时偏振态的示意图（逐个角度单独出图）。

物理模型：入射线偏振光沿 E_x 方向，λ/4 波片快轴与 E_x 夹角 θ。经 1/4 波片后，
电矢量端点轨迹为半轴分别是 cosθ、sinθ 的椭圆。θ=0° 退化为沿 E_x 的线段，
θ=45° 为正圆，θ=90° 退化为沿 E_y 的线段。

两种观察条件对应两套图：
  follows（检偏器随 1/4 玻片转动）—— 参照系跟着波片转，椭圆长轴恒沿 E_x，
      只有长短轴之比随 θ 变化，整个图形不倾斜；
  fixed （检偏器不随 1/4 玻片转动）—— 参照系是实验室坐标系，波片快轴相对
      固定的 E_x 轴转了 θ，出射椭圆的振动主轴随之旋转，图形整体倾斜 θ。

输出：
  qwp_follows_00.png … qwp_follows_90.png   检偏器随玻片转动的 7 张单图
  qwp_fixed_00.png  … qwp_fixed_90.png      检偏器不随玻片转动的 7 张单图
  qwp_column_follows.png / qwp_column_fixed.png   整栏拼版（便于贴进表格）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# 以脚本所在目录为基准，与当前工作目录无关
SCRIPT_DIR = Path(__file__).resolve().parent

RANGE = 1.5  # 横纵坐标统一范围 -1.5 ~ +1.5
TICKS = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]  # 与讲义一致，省去与坐标轴重合的 0
TRACE_COLOR = "darkblue"

NOTE = "$E_y$ 方向为检偏器初始偏振方向"  # 必须显眼标注的说明

# 两种观察条件：键名 -> (是否随波片转动, 表头文字)
CONFIGS = {
    "follows": (True, "检偏器随 1/4 玻片转动"),
    "fixed": (False, "检偏器不随 1/4 玻片转动"),
}

# (θ, 偏振态名称)
CASES = [
    (0, "线偏振光"),
    (15, "椭圆偏振光"),
    (30, "椭圆偏振光"),
    (45, "圆偏振光"),
    (60, "椭圆偏振光"),
    (75, "椭圆偏振光"),
    (90, "线偏振光"),
]


def polarization_trace(theta_deg, follows_analyzer, n=720):
    """返回电矢量端点轨迹 (x, y)，单位为 E0。"""
    th = np.deg2rad(theta_deg)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    ct, st = np.cos(t), np.sin(t)
    if follows_analyzer:
        # 参照系随波片旋转：快轴固定沿 Ex，慢轴固定沿 Ey
        return np.cos(th) * ct, np.sin(th) * st
    # 参照系固定：快轴 u=(cosθ,sinθ)，慢轴 v=(-sinθ,cosθ)
    a, b = np.cos(th), np.sin(th)
    x = a * ct * np.cos(th) - b * st * np.sin(th)
    y = a * ct * np.sin(th) + b * st * np.cos(th)
    return x, y


def draw_state(ax, theta_deg, follows_analyzer):
    """画单个偏振态示意图，坐标轴过原点，刻度标在轴线上（与讲义排版一致）。"""
    ax.set_xlim(-RANGE, RANGE)
    ax.set_ylim(-RANGE, RANGE)
    ax.set_aspect("equal")
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.tick_params(labelsize=8, length=3, width=0.6, pad=1.5)
    # 刻度数字紧贴轴线，会被偏振态轨迹横穿；加白底保证任何情况下都清晰可读
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.4})

    # 坐标轴移到原点，使刻度数字紧贴轴线，不占用绘图区
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Ex / Ey 轴名标在轴的末端，偏移量保证不与刻度数字重叠
    ax.annotate(
        "$E_x$",
        xy=(RANGE, 0.0),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        annotation_clip=False,
    )
    ax.annotate(
        "$E_y$",
        xy=(0.0, RANGE),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        annotation_clip=False,
    )

    x, y = polarization_trace(theta_deg, follows_analyzer)
    ax.plot(x, y, color=TRACE_COLOR, lw=2.2, solid_capstyle="round")


def save_single(theta_deg, kind, follows_analyzer, config_label, filename):
    """单独保存某一个角度的示意图，便于直接贴进实验记录表的单元格。"""
    fig, ax = plt.subplots(figsize=(3.9, 4.5))
    draw_state(ax, theta_deg, follows_analyzer)
    # pad 要够大：顶端刻度 1.5 与 E_y 轴名都压在轴框上沿外，标题再高就撞上了
    ax.set_title(f"$\\theta={theta_deg}^\\circ$   {kind}", fontsize=12, pad=20)

    fig.text(
        0.5,
        0.955,
        config_label,
        ha="center",
        va="top",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.045,
        NOTE,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "black", "lw": 1.3},
    )
    # set_aspect("equal") 的 Axes 与 tight_layout 不兼容，改用固定边距
    fig.subplots_adjust(left=0.19, right=0.90, top=0.845, bottom=0.185)

    out = SCRIPT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def save_sheet(follows_analyzer, config_label, filename):
    """把 7 个角度拼成一版，方便整栏贴进记录表。"""
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, (theta_deg, kind) in zip(axes.ravel(), CASES):
        draw_state(ax, theta_deg, follows_analyzer)
        ax.set_title(f"$\\theta={theta_deg}^\\circ$   {kind}", fontsize=11, pad=20)

    # 第 8 格放必须显眼标注的那句话
    note_ax = axes.ravel()[len(CASES)]
    note_ax.axis("off")
    note_ax.text(
        0.5,
        0.62,
        NOTE,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "ec": "black", "lw": 1.5},
    )
    note_ax.text(
        0.5,
        0.26,
        "横轴 $E_x$，纵轴 $E_y$",
        ha="center",
        va="center",
        fontsize=11.5,
    )

    fig.suptitle(f"λ/4 波片转动时的偏振态（{config_label}）", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.07, wspace=0.20, hspace=0.22)

    out = SCRIPT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    saved = []
    for tag, (follows, label) in CONFIGS.items():
        for theta_deg, kind in CASES:
            saved.append(
                save_single(
                    theta_deg,
                    kind,
                    follows,
                    label,
                    f"qwp_{tag}_{theta_deg:02d}.png",
                )
            )
        saved.append(save_sheet(follows, label, f"qwp_column_{tag}.png"))

    for path in saved:
        print(f"已保存: {path.name}")
    print(f"共 {len(saved)} 个文件。")


if __name__ == "__main__":
    main()
