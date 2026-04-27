from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


OUT = Path("RESS_bearing_RUL_paper_Chinese.pptx")

NAVY = RGBColor(19, 45, 75)
BLUE = RGBColor(45, 104, 196)
LIGHT_BLUE = RGBColor(225, 236, 252)
ORANGE = RGBColor(232, 126, 4)
GREEN = RGBColor(25, 135, 84)
RED = RGBColor(206, 66, 87)
GRAY = RGBColor(92, 101, 116)
LIGHT_GRAY = RGBColor(245, 247, 250)
WHITE = RGBColor(255, 255, 255)


def set_run(run, size=20, bold=False, color=NAVY):
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_textbox(slide, text, x, y, w, h, size=20, bold=False, color=NAVY,
                align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = text
    for run in p.runs:
        set_run(run, size=size, bold=bold, color=color)
    return box


def add_title(slide, title, subtitle=None):
    add_textbox(slide, title, 0.55, 0.28, 12.2, 0.5, size=25, bold=True, color=NAVY)
    line = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0.55), Inches(0.9), Inches(12.2), Inches(0.035)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ORANGE
    line.line.fill.background()
    if subtitle:
        add_textbox(slide, subtitle, 0.58, 0.98, 12, 0.35, size=12, color=GRAY)


def add_footer(slide, page):
    add_textbox(slide, "Deep learning-based RUL estimation of bearings | RESS 182 (2019)", 0.55, 7.05, 8.3, 0.25, size=8.5, color=GRAY)
    add_textbox(slide, f"{page:02d}/15", 12.0, 7.05, 0.8, 0.25, size=8.5, color=GRAY, align=PP_ALIGN.RIGHT)


def add_bullets(slide, items, x, y, w, h, size=17, color=NAVY, bullet_color=ORANGE,
                level_gap=0.22, line_spacing=1.05):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.space_after = Pt(8)
        p.line_spacing = line_spacing
        p.margin_left = Inches(level_gap)
        p.margin_first_line = Inches(-level_gap)
        p.bullet = True
        for run in p.runs:
            set_run(run, size=size, color=color)
    return box


def add_card(slide, x, y, w, h, title, body, accent=BLUE):
    card = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RGBColor(218, 226, 238)
    stripe = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background()
    add_textbox(slide, title, x + 0.18, y + 0.12, w - 0.35, 0.35, size=15, bold=True, color=accent)
    add_textbox(slide, body, x + 0.18, y + 0.58, w - 0.35, h - 0.68, size=13.5, color=NAVY)
    return card


def add_arrow(slide, x1, y1, x2, y2, color=BLUE, width=2.0):
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = color
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def make_bearing_image(path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=180)
    ax.set_aspect("equal")
    ax.axis("off")
    outer = plt.Circle((0, 0), 2.4, fill=False, lw=18, color="#1f4e79", alpha=0.9)
    inner = plt.Circle((0, 0), 1.05, fill=False, lw=16, color="#6c757d", alpha=0.9)
    ax.add_patch(outer)
    ax.add_patch(inner)
    for k in range(12):
        a = 2 * np.pi * k / 12
        x, y = 1.72 * np.cos(a), 1.72 * np.sin(a)
        roller = plt.Circle((x, y), 0.25, color="#f28e2b", ec="#7a3f00", lw=1.5)
        ax.add_patch(roller)
    ax.annotate("radial load", xy=(0, 2.35), xytext=(0, 3.25),
                arrowprops=dict(arrowstyle="->", color="#d1495b", lw=2.5),
                ha="center", color="#d1495b", fontsize=13, weight="bold")
    ax.text(-2.8, -3.0, "Rolling bearing: vibration reflects degradation", fontsize=10, color="#334")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.25, 3.55)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_rul_curve(path):
    rng = np.random.default_rng(7)
    t = np.linspace(0, 100, 400)
    health = 0.12 + 0.0028 * t + 0.000075 * np.maximum(t - 48, 0) ** 2
    health += rng.normal(0, 0.015, t.size)
    threshold = 1.0
    rul = np.maximum(0, 100 - t)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.8), dpi=180, sharex=True)
    axes[0].plot(t, health, color="#2f6bc1", lw=2.2)
    axes[0].axhline(threshold, color="#d1495b", ls="--", lw=1.8)
    axes[0].fill_between(t, 0, health, color="#d7e7ff", alpha=0.7)
    axes[0].set_ylabel("Health indicator")
    axes[0].set_title("Degradation signal and failure threshold", fontsize=11)
    axes[0].text(76, 1.05, "failure threshold", color="#d1495b", fontsize=8)
    axes[1].plot(t, rul, color="#f28e2b", lw=2.5)
    axes[1].fill_between(t, 0, rul, color="#ffe2bd", alpha=0.8)
    axes[1].set_xlabel("Operating time / cycle")
    axes[1].set_ylabel("RUL")
    axes[1].set_title("Remaining useful life target", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_timefreq_image(path):
    rng = np.random.default_rng(10)
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    early = 0.32 * np.sin(2 * np.pi * 45 * t) + 0.06 * rng.normal(size=t.size)
    late = 0.42 * np.sin(2 * np.pi * 45 * t) + 0.18 * np.sin(2 * np.pi * 130 * t)
    late += 0.25 * np.exp(-((t - 0.55) / 0.05) ** 2) * np.sin(2 * np.pi * 260 * t)
    late += 0.09 * rng.normal(size=t.size)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.7), dpi=180)
    axes[0, 0].plot(t, early, lw=1.1, color="#2f6bc1")
    axes[0, 0].set_title("Early-stage vibration")
    axes[1, 0].plot(t, late, lw=1.1, color="#d1495b")
    axes[1, 0].set_title("Degraded vibration")
    axes[0, 1].magnitude_spectrum(early, Fs=fs, color="#2f6bc1", scale="dB")
    axes[0, 1].set_title("Frequency content")
    axes[1, 1].magnitude_spectrum(late, Fs=fs, color="#d1495b", scale="dB")
    axes[1, 1].set_title("Fault-related bands")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_arch_image(path):
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    blocks = [
        (0.2, 1.35, 1.15, 1.0, "vibration\nwindow", "#eaf2ff"),
        (1.8, 1.35, 1.25, 1.0, "time-\nfreq map", "#eaf2ff"),
        (3.6, 2.35, 1.35, 0.75, "small\nkernel", "#d5f2e3"),
        (3.6, 1.35, 1.35, 0.75, "medium\nkernel", "#d5f2e3"),
        (3.6, 0.35, 1.35, 0.75, "large\nkernel", "#d5f2e3"),
        (5.45, 1.35, 1.35, 1.0, "feature\nfusion", "#fff2d9"),
        (7.25, 1.35, 1.1, 1.0, "dense\nlayers", "#fde2e7"),
        (8.85, 1.35, 0.95, 1.0, "RUL", "#fbd1d9"),
    ]
    for x, y, w, h, text, color in blocks:
        rect = plt.Rectangle((x, y), w, h, ec="#42627f", fc=color, lw=1.8, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold", color="#123")
    arrows = [
        ((1.35, 1.85), (1.8, 1.85)),
        ((3.05, 1.85), (3.6, 2.72)),
        ((3.05, 1.85), (3.6, 1.72)),
        ((3.05, 1.85), (3.6, 0.72)),
        ((4.95, 2.72), (5.45, 1.95)),
        ((4.95, 1.72), (5.45, 1.85)),
        ((4.95, 0.72), (5.45, 1.72)),
        ((6.8, 1.85), (7.25, 1.85)),
        ((8.35, 1.85), (8.85, 1.85)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#2f6bc1", lw=2))
    ax.text(0.15, 3.65, "Multi-scale CNN extracts local impulses + broader degradation trends", fontsize=11, color="#2f3e50")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def make_prediction_image(path):
    t = np.linspace(0, 100, 160)
    true = np.maximum(0, 100 - t)
    proposed = np.maximum(0, true + 5 * np.sin(t / 11) * np.exp(-t / 140))
    baseline = np.maximum(0, true + 14 * np.sin(t / 9) + 8)
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    ax.plot(t, true, color="#111827", lw=2.5, label="True RUL")
    ax.plot(t, proposed, color="#2f6bc1", lw=2.2, label="Proposed method")
    ax.plot(t, baseline, color="#f28e2b", lw=1.8, ls="--", label="Typical baseline")
    ax.fill_between(t, proposed - 7, proposed + 7, color="#d7e7ff", alpha=0.55, label="prediction band")
    ax.set_xlabel("Operating time / cycle")
    ax.set_ylabel("RUL")
    ax.set_title("RUL prediction behavior (teaching schematic)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_reliability_loop(path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    nodes = [
        (1.4, 4.6, "sensor\nmonitoring", "#eaf2ff"),
        (5.0, 4.6, "RUL\nprediction", "#d5f2e3"),
        (8.3, 3.0, "maintenance\ndecision", "#fff2d9"),
        (5.0, 1.2, "design\nfeedback", "#fde2e7"),
        (1.4, 3.0, "failure data\n& tests", "#e9ecef"),
    ]
    for x, y, text, color in nodes:
        circ = plt.Circle((x, y), 0.78, color=color, ec="#42627f", lw=2)
        ax.add_patch(circ)
        ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold", color="#123")
    arrow_pairs = [((2.1, 4.6), (4.2, 4.6)), ((5.8, 4.35), (7.65, 3.35)),
                   ((7.65, 2.55), (5.7, 1.55)), ((4.25, 1.35), (1.95, 2.45)),
                   ((1.4, 3.8), (1.4, 4.0))]
    for a, b in arrow_pairs:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#2f6bc1", lw=2))
    ax.text(0.6, 5.6, "Closed-loop reliability design and PHM", fontsize=13, weight="bold", color="#1f4e79")
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


def set_background(slide, color=RGBColor(250, 252, 255)):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def build_ppt(img):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_textbox(slide, "论文精读：滚动轴承剩余寿命预测", 0.7, 0.7, 7.4, 0.6, size=28, bold=True, color=NAVY)
    add_textbox(slide, "Deep learning-based remaining useful life estimation of bearings using multi-scale feature extraction", 0.72, 1.55, 7.6, 0.95, size=18, bold=True, color=BLUE)
    add_textbox(slide, "Xiang Li, Wei Zhang, Qian Ding\nReliability Engineering & System Safety, 182 (2019), 208–218\nDOI: https://doi.org/10.1016/j.ress.2018.11.011", 0.75, 2.75, 7.15, 1.3, size=15.5, color=NAVY)
    slide.shapes.add_picture(str(img["bearing"]), Inches(8.0), Inches(0.95), Inches(4.6), Inches(4.6))
    add_textbox(slide, "面向机械可靠性设计：用振动信号提前估计轴承还能可靠工作多久", 0.78, 5.75, 11.7, 0.55, size=18, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    add_footer(slide, 1)

    # 2
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "1. 论文信息与阅读主线", "本页只交代论文对象；不展开期刊介绍。")
    add_card(slide, 0.7, 1.35, 4.0, 4.5, "论文对象", "研究对象：滚动轴承\n任务：剩余寿命 RUL 预测\n数据：PRONOSTIA 轴承加速退化实验\n方法：时频信息 + 多尺度 CNN 特征提取\n输出：每个监测时刻的 RUL 估计", accent=BLUE)
    add_card(slide, 4.95, 1.35, 3.7, 4.5, "为什么适合机械可靠性设计", "轴承是旋转机械关键件；故障会引发停机、二次损伤与安全风险。\n\n论文将“故障后诊断”推进到“故障前寿命预测”，可直接服务于预测性维护和可靠性闭环设计。", accent=ORANGE)
    add_card(slide, 8.9, 1.35, 3.45, 4.5, "本 PPT 讲解路径", "问题背景\nRUL 与退化数据\n多尺度深度学习模型\n实验验证与结果\n对可靠性设计的启示\n局限与可改进方向", accent=GREEN)
    add_footer(slide, 2)

    # 3
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "2. 研究问题：从“能不能用”到“还能用多久”")
    add_bullets(slide, [
        "传统故障诊断回答：轴承当前是否已经异常、属于哪类故障。",
        "可靠性设计和维护更关心：在当前退化状态下，距离失效还有多少寿命余量。",
        "RUL 预测把传感器监测数据转化为可执行的维护窗口，减少过早维修和突发停机。",
        "难点在于轴承退化高度非线性：冲击、噪声、工况变化和个体差异都会干扰寿命估计。",
    ], 0.78, 1.25, 6.0, 4.6, size=17.5)
    add_card(slide, 7.15, 1.35, 4.95, 1.2, "诊断", "当前是否失效？故障类型是什么？", accent=GRAY)
    add_arrow(slide, 9.55, 2.65, 9.55, 3.38, color=ORANGE)
    add_card(slide, 7.15, 3.45, 4.95, 1.45, "预测", "未来何时达到失效阈值？\n应在何时检修或更换？", accent=ORANGE)
    add_arrow(slide, 9.55, 5.0, 9.55, 5.62, color=GREEN)
    add_card(slide, 7.15, 5.68, 4.95, 0.85, "可靠性设计", "把寿命数据反馈到设计与维护策略。", accent=GREEN)
    add_footer(slide, 3)

    # 4
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "3. RUL 概念：退化轨迹到失效阈值的时间距离")
    slide.shapes.add_picture(str(img["rul"]), Inches(0.85), Inches(1.28), Inches(6.35), Inches(4.55))
    add_bullets(slide, [
        "RUL（Remaining Useful Life）= 当前时刻到失效时刻的剩余运行时间或循环数。",
        "退化指标可以来自振动幅值、频带能量、时频图像或深度网络自动学习的特征。",
        "当退化指标穿越失效阈值时，认为轴承达到寿命终点。",
        "预测越靠近真实 RUL，维修计划越能兼顾安全冗余和经济性。",
    ], 7.55, 1.35, 4.8, 4.6, size=17)
    add_footer(slide, 4)

    # 5
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "4. 数据基础：PRONOSTIA 轴承加速退化实验")
    slide.shapes.add_picture(str(img["bearing"]), Inches(0.8), Inches(1.35), Inches(4.2), Inches(4.2))
    add_bullets(slide, [
        "论文在公开、常用的 PRONOSTIA 滚动轴承数据集上验证方法。",
        "实验记录轴承从健康运行到失效的全寿命振动信号，适合构造监督学习样本。",
        "不同载荷/转速工况使数据更接近工程场景：同类轴承也可能呈现不同退化速度。",
        "机械含义：振动信号中的冲击、频带能量和非平稳成分，是滚道/滚动体损伤扩展的外在表征。",
    ], 5.35, 1.28, 6.7, 4.95, size=17)
    add_footer(slide, 5)

    # 6
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "5. 总体技术路线：原始振动信号到 RUL 估计")
    labels = [
        ("振动采集", "加速度传感器\n连续采样", 0.75),
        ("样本切片", "按时间窗口\n构造训练样本", 2.75),
        ("时频表达", "同时保留\n时间与频率信息", 4.75),
        ("多尺度 CNN", "不同卷积尺度\n提取退化特征", 6.75),
        ("RUL 回归", "输出剩余寿命\n用于维护决策", 8.95),
    ]
    for title, body, x in labels:
        add_card(slide, x, 2.25, 1.55, 1.85, title, body, accent=BLUE if x < 6 else ORANGE)
    for x in [2.32, 4.32, 6.32, 8.52]:
        add_arrow(slide, x, 3.15, x + 0.35, 3.15, color=ORANGE)
    add_bullets(slide, [
        "关键思想：不依赖大量手工特征，直接让深度网络从时频数据中学习退化模式。",
        "多尺度结构使模型能同时关注局部冲击和较长时间范围的趋势变化。",
    ], 1.15, 5.0, 10.8, 1.0, size=17)
    add_footer(slide, 6)

    # 7
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "6. 为什么要看时频信息？")
    slide.shapes.add_picture(str(img["timefreq"]), Inches(0.72), Inches(1.18), Inches(6.9), Inches(4.85))
    add_bullets(slide, [
        "轴承早期损伤往往表现为弱冲击，直接从时域幅值中识别并不稳定。",
        "频域能揭示故障相关频带和调制成分，但会弱化冲击出现的时间位置。",
        "时频表达把“什么时候出现异常”和“异常集中在哪些频带”结合起来。",
        "这为 CNN 的二维局部特征提取提供了类似图像的输入结构。",
    ], 8.0, 1.35, 4.35, 4.6, size=16.5)
    add_textbox(slide, "图为教学示意，非论文原图", 0.9, 6.15, 5.5, 0.25, size=9, color=GRAY)
    add_footer(slide, 7)

    # 8
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "7. 方法核心：多尺度 CNN 特征提取")
    slide.shapes.add_picture(str(img["arch"]), Inches(0.95), Inches(1.15), Inches(7.25), Inches(3.35))
    add_bullets(slide, [
        "不同卷积核/感受野相当于从不同“观察尺度”审视退化信号。",
        "小尺度：更敏感于局部冲击、尖峰与短时异常。",
        "中/大尺度：更适合捕捉频带能量变化和较慢的退化趋势。",
        "多尺度特征融合后，再通过回归层输出 RUL。",
    ], 8.35, 1.25, 4.05, 3.55, size=16.5)
    add_card(slide, 1.0, 5.05, 11.25, 0.95, "与传统特征工程的区别", "传统方法常需人工选择 RMS、峭度、包络谱等指标；论文方法把特征学习嵌入模型训练，降低对专家手工调参的依赖。", accent=GREEN)
    add_footer(slide, 8)

    # 9
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "8. 监督学习构造：寿命标签、窗口样本与模型训练")
    add_card(slide, 0.8, 1.35, 3.45, 4.55, "样本", "从全寿命振动序列中截取多个时间窗口。\n\n每个窗口代表某个运行时刻的健康状态。", accent=BLUE)
    add_card(slide, 4.9, 1.35, 3.45, 4.55, "标签", "根据该时刻到失效终点的距离生成 RUL 标签。\n\n越接近失效，标签值越小。", accent=ORANGE)
    add_card(slide, 9.0, 1.35, 3.45, 4.55, "训练目标", "让网络学习“信号模式 → 剩余寿命”的映射。\n\n测试时输入新轴承监测片段，即可输出 RUL。", accent=GREEN)
    add_arrow(slide, 4.25, 3.62, 4.9, 3.62, color=ORANGE)
    add_arrow(slide, 8.35, 3.62, 9.0, 3.62, color=ORANGE)
    add_footer(slide, 9)

    # 10
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "9. 实验验证关注点：预测精度与工程可用性")
    add_bullets(slide, [
        "数据划分：用部分轴承退化轨迹训练，在其他轴承轨迹上测试泛化能力。",
        "比较对象：传统数据驱动方法、单一尺度/较浅模型等基线方案。",
        "评价重点：预测误差是否更小、曲线是否更平滑、临近失效阶段是否仍保持可靠。",
        "工程可用性：模型是否减少人工特征依赖，并能在实时监测中持续更新 RUL。",
    ], 0.85, 1.35, 5.8, 4.7, size=17)
    metrics = [
        ("误差", "预测值与真实 RUL 的偏差"),
        ("稳定性", "预测曲线是否剧烈跳动"),
        ("提前量", "是否能在失效前给出有效预警"),
        ("泛化", "不同轴承/工况下是否仍可用"),
    ]
    for i, (m, d) in enumerate(metrics):
        add_card(slide, 7.1, 1.25 + i * 1.22, 4.65, 0.9, m, d, accent=[BLUE, ORANGE, GREEN, RED][i])
    add_footer(slide, 10)

    # 11
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "10. 主要结果：多尺度深度特征提高 RUL 预测表现")
    slide.shapes.add_picture(str(img["prediction"]), Inches(0.8), Inches(1.25), Inches(6.65), Inches(3.9))
    add_bullets(slide, [
        "论文报告：所提多尺度深度学习方法在 PRONOSTIA 轴承数据上取得较高 RUL 预测精度。",
        "相较依赖手工特征或单一模型的方案，多尺度 CNN 能更充分挖掘退化信息。",
        "模型输出的 RUL 曲线更接近真实寿命趋势，对预测性维护更有参考价值。",
        "右图为教学示意：强调预测曲线与真实 RUL 的贴合关系，并非论文数值复刻。",
    ], 7.8, 1.3, 4.7, 4.6, size=16.2)
    add_footer(slide, 11)

    # 12
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "11. 对可靠性设计的价值：把监测数据变成决策量")
    slide.shapes.add_picture(str(img["loop"]), Inches(0.8), Inches(1.25), Inches(6.75), Inches(3.75))
    add_bullets(slide, [
        "RUL 不是单纯算法输出，而是连接“状态监测—维修决策—设计改进”的桥梁。",
        "当 RUL 低于维护阈值，可安排检修、更换或降载运行，避免连锁故障。",
        "多台设备的 RUL 分布可帮助制定备件库存与维修资源配置。",
        "寿命预测误差和失效样本还能反向暴露结构、材料、润滑或装配的薄弱环节。",
    ], 7.85, 1.25, 4.6, 4.8, size=16.3)
    add_footer(slide, 12)

    # 13
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "12. 机械可靠性视角：与失效机理如何对应？")
    add_card(slide, 0.8, 1.25, 3.55, 4.8, "物理退化", "接触疲劳\n滚道点蚀/剥落\n滚动体损伤\n润滑劣化\n装配偏差与载荷冲击", accent=BLUE)
    add_card(slide, 4.9, 1.25, 3.55, 4.8, "可观测信号", "振动幅值增加\n冲击脉冲增多\n频带能量迁移\n时频图纹理变化\n退化趋势非线性", accent=ORANGE)
    add_card(slide, 9.0, 1.25, 3.55, 4.8, "可靠性量化", "健康指标\n失效阈值\nRUL 分布或点估计\n维护阈值\n设计安全裕度", accent=GREEN)
    add_arrow(slide, 4.35, 3.65, 4.9, 3.65, color=ORANGE)
    add_arrow(slide, 8.45, 3.65, 9.0, 3.65, color=ORANGE)
    add_footer(slide, 13)

    # 14
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "13. 局限与可改进方向")
    add_bullets(slide, [
        "需要较充分的带标签全寿命退化数据；真实工厂中完整失效样本通常稀缺。",
        "不同设备、载荷、转速、传感器安装位置会造成数据分布偏移，影响模型迁移。",
        "深度模型可解释性有限：预测准确不等于清楚知道具体失效机理。",
        "若只给出点估计 RUL，难以直接表达置信区间和风险水平。",
        "后续可结合物理退化模型、迁移学习、不确定性量化和在线更新。",
    ], 0.85, 1.25, 6.0, 4.9, size=16.8)
    add_card(slide, 7.25, 1.35, 4.65, 1.15, "给机械研究生的思考", "可靠性设计不应把深度学习当黑盒终点，而应把它作为连接监测数据与失效机理的工具。", accent=RED)
    add_card(slide, 7.25, 3.05, 4.65, 1.15, "工程落地重点", "模型需要在目标设备上校准阈值、验证误报/漏报风险，并与维修制度联动。", accent=ORANGE)
    add_card(slide, 7.25, 4.75, 4.65, 1.15, "研究延伸", "可加入置信区间、寿命分布、成本函数和可靠性约束优化。", accent=GREEN)
    add_footer(slide, 14)

    # 15
    slide = prs.slides.add_slide(blank)
    set_background(slide)
    add_title(slide, "14. 总结：这篇论文给机械可靠性设计的三点启示")
    add_card(slide, 0.85, 1.35, 3.75, 2.65, "启示一", "轴承可靠性评估可以从“离线寿命试验”扩展到“在线状态预测”。", accent=BLUE)
    add_card(slide, 4.85, 1.35, 3.75, 2.65, "启示二", "时频信息 + 多尺度特征有助于捕捉复杂非线性退化过程。", accent=ORANGE)
    add_card(slide, 8.85, 1.35, 3.75, 2.65, "启示三", "RUL 预测的最终价值在于维护决策、风险控制和设计反馈。", accent=GREEN)
    add_textbox(slide, "论文标题", 0.9, 4.55, 1.4, 0.3, size=14, bold=True, color=GRAY)
    add_textbox(slide, "Deep learning-based remaining useful life estimation of bearings using multi-scale feature extraction", 2.2, 4.48, 9.9, 0.45, size=16, bold=True, color=NAVY)
    add_textbox(slide, "论文链接", 0.9, 5.22, 1.4, 0.3, size=14, bold=True, color=GRAY)
    add_textbox(slide, "https://doi.org/10.1016/j.ress.2018.11.011\nhttps://www.sciencedirect.com/science/article/pii/S0951832018308299", 2.2, 5.12, 9.9, 0.75, size=14.5, color=BLUE)
    add_textbox(slide, "参考来源：论文摘要、Highlights、DOI/ScienceDirect/INIS/IDEAS 元数据；PPT 中图示为教学重绘。", 0.9, 6.25, 11.7, 0.35, size=10, color=GRAY, align=PP_ALIGN.CENTER)
    add_footer(slide, 15)

    prs.save(OUT)
    return OUT


def main():
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img = {
            "bearing": tmp / "bearing.png",
            "rul": tmp / "rul.png",
            "timefreq": tmp / "timefreq.png",
            "arch": tmp / "arch.png",
            "prediction": tmp / "prediction.png",
            "loop": tmp / "loop.png",
        }
        make_bearing_image(img["bearing"])
        make_rul_curve(img["rul"])
        make_timefreq_image(img["timefreq"])
        make_arch_image(img["arch"])
        make_prediction_image(img["prediction"])
        make_reliability_loop(img["loop"])
        out = build_ppt(img)
    print(f"Created {out.resolve()}")


if __name__ == "__main__":
    main()
