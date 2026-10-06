"""Render the code-aligned TACM paper figure; no project/model imports required.

Run from any directory: python tools/render_tacm_architecture.py
Requires only matplotlib. SVG text remains editable and PDF uses embedded fonts.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
INK = "#263746"
BLUE = "#397397"
ORANGE = "#AF7448"
GREEN = "#528273"
GRAY = "#71808C"
FILLS = {"blue": "#EAF2F8", "orange": "#FCF0E4", "green": "#EBF4EF", "gray": "#F2F4F6"}


def label(ax, x, y, text, size=14, color=INK, weight="normal", ha="center"):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=ha, va="center", linespacing=1.2, zorder=5)


def box(ax, x, y, w, h, text, kind="blue", size=14, dashed=False):
    edge = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "gray": GRAY}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.55",
                              fc=FILLS[kind], ec=edge, lw=1.25,
                              linestyle="--" if dashed else "-", zorder=3))
    label(ax, x + w / 2, y + h / 2, text, size)


def arrow(ax, points, style="solid", color=BLUE, width=1.4):
    """Orthogonal routes with an arrowhead only on the final segment."""
    linestyle = {"solid": "-", "supervision": "--", "temporal": ":"}[style]
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=width, ls=linestyle, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                               mutation_scale=13, lw=width, linestyle=linestyle,
                               color=color, shrinkA=0, shrinkB=1, zorder=2))


def training_region(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.7",
                              fc="#FCFCFD", ec="#BCC6CE", lw=1.2, ls="--", zorder=0))


def framework(ax):
    label(ax, 2, 54, "(a)  整体框架：集中训练、分散执行", 18, weight="bold", ha="left")
    box(ax, 2, 33, 27, 16, "环境与轨迹采集\nv3.10 异构空战\n红方：1 MAV + 3 UAV\n蓝方：4 架飞机\nMAV 无直接攻击能力", size=13)
    ax.add_patch(FancyBboxPatch((37, 38), 104, 11.3, boxstyle="round,pad=0.1,rounding_size=0.5",
                              fc="white", ec="#CBDCE7", lw=1, zorder=1))
    box(ax, 39, 42, 25, 6, "MAV 策略网络\n高斯多层感知机", size=14)
    for x, aid in ((70, "UAV1"), (94, "UAV2"), (118, "UAV3")):
        box(ax, x, 42, 22, 6, f"{aid} 策略网络\nTACM", size=14)
    label(ax, 91, 39.8, "三架 UAV：结构相同、参数独立", 13, BLUE)
    label(ax, 90, 36.5, "执行阶段仅使用局部观测与策略网络，不使用价值网络、教师或辅助损失", 13, weight="bold")
    arrow(ax, [(29, 45), (39, 45)])
    label(ax, 34, 47.8, r"$o_t^M$", 14)
    arrow(ax, [(29, 42), (33, 42), (33, 50.4), (129, 50.4), (129, 48)])
    arrow(ax, [(81, 50.4), (81, 48)])
    arrow(ax, [(105, 50.4), (105, 48)])
    label(ax, 83, 52.1, r"局部观测 $o_t^1, o_t^2, o_t^3$", 14)
    # Four action outputs join a return bus below the actor row.
    for x in (51.5, 81, 105, 129):
        arrow(ax, [(x, 42), (x, 41)])
    ax.plot([35, 135], [41, 41], color=BLUE, lw=1.4, zorder=2)
    arrow(ax, [(35, 41), (35, 35), (29, 35)])
    label(ax, 17, 31, r"三维连续动作：$a_t^M,a_t^1,a_t^2,a_t^3$", 12)

    training_region(ax, 2, 1, 138, 27.5)
    label(ax, 5, 27, "以下模块仅用于集中训练", 12, GRAY, weight="bold", ha="left")
    arrow(ax, [(15, 33), (15, 29)])
    box(ax, 4, 20, 25, 5, "全局状态 $s_t$\n共享团队奖励", "gray", 13)
    box(ax, 34, 20, 29, 5, "集中式团队价值网络\n$V_{team}(s_t)$", "gray", 13)
    box(ax, 68, 20, 20, 5, "团队 GAE\n$A_{team}$", "gray", 13)
    arrow(ax, [(29, 24), (34, 24)], color=GRAY)
    arrow(ax, [(29, 21), (31.5, 21), (31.5, 26), (78, 26), (78, 25)], color=GRAY)
    arrow(ax, [(63, 22.5), (68, 22.5)], color=GRAY)
    box(ax, 4, 11, 25, 7, "局部观测\n辅助奖励 = 过程奖励\n+ 仅本机损失事件", "orange", 12.5)
    box(ax, 34, 15, 29, 3, "MAV 辅助价值网络  $V_{aux,M}(o^M)$", "orange", 11.5)
    box(ax, 34, 11, 29, 3, "UAV 共享辅助价值网络  $V_{aux,U}(o^i)$", "orange", 10.5)
    label(ax, 48.5, 9.7, "一个价值网络，由 UAV1/2/3 共享", 11.5, ORANGE)
    box(ax, 68, 11, 20, 7, "辅助 GAE\n个体死亡时截断\n$A_{aux,i}$", "orange", 12)
    arrow(ax, [(29, 16.5), (34, 16.5)], color=ORANGE)
    arrow(ax, [(29, 12.5), (34, 12.5)], color=ORANGE)
    arrow(ax, [(63, 16.5), (68, 16.5)], color=ORANGE)
    arrow(ax, [(63, 12.5), (68, 12.5)], color=ORANGE)
    arrow(ax, [(29, 14.5), (31, 14.5), (31, 18.9), (78, 18.9), (78, 18)], color=ORANGE)
    box(ax, 94, 13, 21, 12, "优势融合\n$A_i=\\mathrm{Norm}(A_{team})$\n$+\\,0.5\\,\\mathrm{Norm}(A_{aux,i})$\n融合后不再归一化", "green", 11)
    arrow(ax, [(88, 22.5), (94, 22.5)], color=GRAY)
    arrow(ax, [(88, 14.5), (94, 14.5)], color=ORANGE)
    box(ax, 120, 13, 18, 12, "HAPPO 顺序更新\n随机智能体顺序\n前序修正因子 $F_i$\nPPO 使用 $F_i A_i$", "green", 11.5)
    arrow(ax, [(115, 19), (120, 19)], color=GREEN)
    arrow(ax, [(129, 25), (129, 32), (37, 32), (37, 39)], "supervision", GREEN)
    label(ax, 92, 33.5, "更新四个独立策略网络", 12, GREEN)

    box(ax, 4, 2.3, 43, 6, "战术上下文教师：$s_t$ → $q_{teacher}$ 与置信度\n团队可见敌机 / 交战准备度\nMAV 威胁 / 支援责任", "orange", 11.5)
    # Explicit state input to teacher, outside the role branch.
    arrow(ax, [(4, 24), (3, 24), (3, 5.3), (4, 5.3)], "supervision", ORANGE)
    box(ax, 53, 2.3, 38, 6, "事件感知时序一致性\n相邻路由概率 + 事件掩码\n击杀/死亡；目标与教师模式稳定", "orange", 11)
    box(ax, 97, 2.3, 41, 6, "仅对 UAV 路由器施加正则化\n置信度加权上下文 KL + 时序 L2\n编码特征停止梯度；仅训练时使用", "orange", 11)
    arrow(ax, [(47, 5.3), (50, 5.3), (50, 1.5), (118, 1.5), (118, 2.3)], "supervision", ORANGE)
    arrow(ax, [(91, 5.3), (97, 5.3)], "temporal", ORANGE)
    arrow(ax, [(138, 5.3), (143, 5.3), (143, 39), (141, 39)], "supervision", ORANGE)


def actor_detail(ax):
    label(ax, 2, 54, "(b)  单 UAV 策略网络：软战术模式路由", 18, weight="bold", ha="left")
    box(ax, 2, 26, 15, 9, "局部观测\n100 维", size=14)
    box(ax, 23, 26, 23, 9, "本机分支共享编码器\n100 → 128 → 128\n每层：线性变换 + Tanh", size=11.5)
    arrow(ax, [(17, 30.5), (23, 30.5)])
    arrow(ax, [(46, 30.5), (51, 30.5)])
    ax.plot([51, 51], [17, 44], color=BLUE, lw=1.4, zorder=2)
    label(ax, 49, 36.5, "$h_i$", 14)
    box(ax, 57, 40, 25, 8, "基础均值头\n线性变换 128 → 3\n$\mu_{base}$", size=14)
    arrow(ax, [(51, 44), (57, 44)])
    box(ax, 57, 27, 25, 9, "软路由器\n线性 128 → 2 + Softmax\n$p_{eng},\ p_{sup}$", "orange", 12)
    arrow(ax, [(51, 31.5), (57, 31.5)])
    box(ax, 57, 12, 12, 10, "交战导向\n残差专家\n线性 128 → 3\nTanh", "green", 10.5)
    box(ax, 72, 12, 12, 10, "支援导向\n残差专家\n线性 128 → 3\nTanh", "green", 10.5)
    arrow(ax, [(51, 17), (57, 17)])
    arrow(ax, [(51, 17), (51, 23.8), (78, 23.8), (78, 22)])
    box(ax, 91, 23, 24, 12, "软残差混合\n$0.25\,(p_{eng}e_{eng}$\n$+\;p_{sup}e_{sup})$\n非离散模式硬切换", "green", 13)
    arrow(ax, [(82, 31.5), (91, 31.5)], color=ORANGE)
    arrow(ax, [(63, 12), (63, 10), (86, 10), (86, 26), (91, 26)], color=GREEN)
    arrow(ax, [(84, 19), (88, 19), (88, 25), (91, 25)], color=GREEN)
    ax.add_patch(Circle((120, 44), 2, fc="white", ec=BLUE, lw=1.5, zorder=3))
    label(ax, 120, 44, "+", 20)
    arrow(ax, [(82, 44), (118, 44)])
    arrow(ax, [(115, 29), (120, 29), (120, 42)], color=GREEN)
    label(ax, 108, 39, "最终均值 $\mu_i$", 13)
    box(ax, 91, 49, 35, 4, "可训练 log-std 参数（3 维、与状态无关）", "gray", 12)
    box(ax, 128, 31, 14, 15, "高斯策略\n$\mathcal{N}(\mu_i,\sigma_i^2)$\n采样 $z_i$\n或使用均值", size=13)
    arrow(ax, [(122, 44), (128, 44)])
    arrow(ax, [(126, 51), (135, 51), (135, 46)], color=GRAY)
    box(ax, 128, 14, 14, 9, "Tanh 压缩\n$a_i=\tanh(z_i)$\n三维动作", size=13)
    arrow(ax, [(135, 31), (135, 23)])
    label(ax, 108, 11, "各 UAV 网络参数独立", 12, BLUE)

    training_region(ax, 2, 1, 140, 8)
    box(ax, 3, 2, 22, 6, "教师监督目标\n$q_{teacher}$ 与置信度", "orange", 12, True)
    box(ax, 31, 2, 50, 6, "上下文监督：置信度加权 KL\n$h_i$ 停止梯度，仅直接更新路由器", "orange", 12, True)
    box(ax, 91, 2, 50, 6, "事件感知时序 L2：仅作用于路由器\n前一时刻的概率分布停止梯度", "orange", 12, True)
    arrow(ax, [(25, 5), (31, 5)], "supervision", ORANGE)
    arrow(ax, [(44, 8), (48, 8), (48, 25), (56, 25), (56, 29), (57, 29)], "supervision", ORANGE)
    arrow(ax, [(106, 8), (87, 8), (87, 37.5), (70, 37.5), (70, 36)], "temporal", ORANGE)
    label(ax, 9, 20, "辅助损失仅用于训练\n推理时不使用", 12, ORANGE, ha="left")


def chinese_font():
    """Use an installed CJK font; never silently render missing Chinese glyphs."""
    for family in ("Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "SimHei", "WenQuanYi Zen Hei"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        print(f"Chinese font: {family} ({path})")
        return family
    # WSL can reuse Windows fonts without installing a new dependency.
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("/mnt/c/Windows/Fonts/msyh.ttc")):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    raise RuntimeError("Chinese font unavailable: install Noto Sans CJK SC or Microsoft YaHei, then rerun.")


def main():
    plt.rcParams.update({"font.family": [chinese_font(), "DejaVu Sans"], "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(14.4, 11.7), facecolor="white")
    fig.text(0.5, 0.981, "TACM：战术上下文感知的一致性模式学习",
             ha="center", va="top", fontsize=21, weight="bold", color=INK)
    for rect, draw in (((0.015, 0.505, 0.97, 0.435), framework),
                       ((0.015, 0.052, 0.97, 0.435), actor_detail)):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 145)
        ax.set_ylim(0, 57)
        ax.set_axis_off()
        draw(ax)
    legend = fig.add_axes((0.025, 0.005, 0.95, 0.036))
    legend.set_xlim(0, 145); legend.set_ylim(0, 4); legend.set_axis_off()
    for x, style, color, text in ((1, "solid", BLUE, "实线：执行 / 前向数据流"),
                                (48, "supervision", ORANGE, "虚线：训练期监督 / 参数更新"),
                                (106, "temporal", ORANGE, "点线：时序一致性正则化")):
        arrow(legend, [(x, 2), (x + 5, 2)], style, color)
        label(legend, x + 6, 2, text, 12, ha="left")
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = OUT / f"tacm_architecture.{extension}"
        fig.savefig(path, dpi=350, facecolor="white", metadata={"Creator": "render_tacm_architecture.py"} if extension == "pdf" else None)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
