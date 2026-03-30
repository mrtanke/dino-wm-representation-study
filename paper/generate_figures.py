from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "docs" / "EXPERIMENT_TRACKER.csv"
FIG_DIR = ROOT / "paper" / "figure"


def label_group(encoder: str, transition: str) -> str:
    enc_map = {
        "dino": "Patch",
        "dino_cls": "CLS",
    }
    trans_map = {
        "deterministic": "Det",
        "gaussian": "Gauss",
    }
    return f"{enc_map[encoder]}-{trans_map[transition]}"


def save_formal_matrix(df: pd.DataFrame) -> None:
    # Formal matrix rows are tagged as F0_train/F0_plan, F1_train/F1_plan, etc.
    plan_rows = df[df["experiment_id"].str.match(r"F\d+_plan$")].copy()
    train_rows = df[df["experiment_id"].str.match(r"F\d+_train$")].copy()

    plan_rows["group"] = [
        label_group(e, t) for e, t in zip(plan_rows["encoder"], plan_rows["transition"])
    ]
    train_rows["group"] = [
        label_group(e, t) for e, t in zip(train_rows["encoder"], train_rows["transition"])
    ]

    state_dist = (
        plan_rows.groupby("group", sort=False)["final_goal_distance"].mean().reindex(
            ["Patch-Det", "CLS-Det", "Patch-Gauss", "CLS-Gauss"]
        )
    )
    train_loss = (
        train_rows.groupby("group", sort=False)["train_loss"].mean().reindex(
            ["Patch-Det", "CLS-Det", "Patch-Gauss", "CLS-Gauss"]
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f"]

    axes[0].bar(state_dist.index, state_dist.values, color=colors)
    axes[0].set_title("Formal Matrix Mean State Distance")
    axes[0].set_ylabel("Mean state distance")
    axes[0].tick_params(axis="x", rotation=20)

    y_positions = list(range(len(train_loss.index)))
    axes[1].axvline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.8)
    axes[1].scatter(train_loss.values, y_positions, s=110, c=colors, zorder=3)
    for y, value in zip(y_positions, train_loss.values):
        x_offset = 0.08 if value >= 0 else -0.08
        ha = "left" if value >= 0 else "right"
        axes[1].text(value + x_offset, y, f"{value:.2f}", va="center", ha=ha, fontsize=10)
    axes[1].set_yticks(y_positions, train_loss.index)
    axes[1].set_title("Formal Matrix Mean Training Loss")
    axes[1].set_xlabel("Mean train loss")
    axes[1].set_ylim(-0.5, len(y_positions) - 0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "formal_matrix_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_patch_large_eval(df: pd.DataFrame) -> None:
    records = []
    records.append({"seed": "Seed 0", "transition": "Det", "dist": 3.1738})
    records.append({"seed": "Seed 0", "transition": "Gauss", "dist": (4.4740 + 4.2024) / 2})
    records.append({"seed": "Seed 1", "transition": "Det", "dist": (4.0346 + 4.2033) / 2})
    records.append({"seed": "Seed 1", "transition": "Gauss", "dist": (3.2888 + 4.5844) / 2})
    records.append({"seed": "Seed 2", "transition": "Det", "dist": (3.5163 + 3.4006) / 2})
    records.append({"seed": "Seed 2", "transition": "Gauss", "dist": (3.4006 + 5.3276) / 2})
    plot_df = pd.DataFrame(records)

    order = ["Seed 0", "Seed 1", "Seed 2"]
    det = plot_df[plot_df["transition"] == "Det"].set_index("seed").loc[order]["dist"]
    gau = plot_df[plot_df["transition"] == "Gauss"].set_index("seed").loc[order]["dist"]

    x = range(len(order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width / 2 for i in x], det.values, width, label="Patch-Det", color="#355070")
    ax.bar([i + width / 2 for i in x], gau.values, width, label="Patch-Gauss", color="#e56b6f")
    ax.set_xticks(list(x), order)
    ax.set_ylabel("Mean state distance")
    ax.set_title("Patch Branch Larger-Sample Follow-up")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(FIG_DIR / "patch_large_eval.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_large_eval_summary(df: pd.DataFrame) -> None:
    records = [
        ("Patch-Det S0", 3.1738, "complete"),
        ("CLS-Det S0", (3.7136 + 3.3445) / 2, "complete"),
        ("Patch-Gauss S0", (4.4740 + 4.2024) / 2, "complete"),
        ("CLS-Gauss S0", (4.2965 + 3.3520) / 2, "complete"),
        ("Patch-Det S1", (4.0346 + 4.2033) / 2, "complete"),
        ("CLS-Det S1", 3.9672, "partial"),
        ("Patch-Gauss S1", (3.2888 + 4.5844) / 2, "complete"),
        ("CLS-Gauss S1", (3.8905 + 4.0224) / 2, "complete"),
        ("Patch-Det S2", (3.5163 + 3.4006) / 2, "complete"),
        ("Patch-Gauss S2", (3.4006 + 5.3276) / 2, "complete"),
    ]
    plot_df = pd.DataFrame(records, columns=["label", "dist", "coverage"])
    colors = ["#355070" if c == "complete" else "#eaac8b" for c in plot_df["coverage"]]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(plot_df["label"], plot_df["dist"], color=colors)
    for bar, value in zip(bars, plot_df["dist"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Mean state distance")
    ax.set_title("Usable Larger-Sample PointMaze Summary")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_eval_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cross_env(df: pd.DataFrame) -> None:
    records = [
        ("PointMaze pretrained", 0.9909),
        ("Wall base", 1.7964),
        ("Wall stronger", 4.1456),
        ("PushT official-like", 20.7644),
    ]
    labels = [x[0] for x in records]
    values = [x[1] for x in records]
    colors = ["#355070", "#6d597a", "#b56576", "#eaac8b"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean state distance")
    ax.set_title("Completed Cross-Environment Pretrained Planning Runs")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_env_sanity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_exploratory_encoders(df: pd.DataFrame) -> None:
    records = [
        ("DINOv2", 5.100910319077759, 1.0),
        ("VFM-VAE", 4.240706682351348, 0.9375),
        ("V-JEPA2", 4.302196876673046, 0.6250),
    ]
    labels = [x[0] for x in records]
    state_dist = [x[1] for x in records]
    success = [x[2] for x in records]
    colors = ["#355070", "#b56576", "#6d597a"]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    bars = ax1.bar(labels, state_dist, color=colors)
    for bar, value in zip(bars, state_dist):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax1.set_ylabel("Mean state distance")
    ax1.set_title("Exploratory PointMaze Encoder Comparison")

    ax2 = ax1.twinx()
    ax2.plot(labels, success, color="#222222", marker="o", linewidth=1.8)
    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
    for x, value in zip(labels, success):
        ax2.text(x, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exploratory_encoder_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRACKER)
    save_formal_matrix(df)
    save_patch_large_eval(df)
    save_large_eval_summary(df)
    save_cross_env(df)
    save_exploratory_encoders(df)


if __name__ == "__main__":
    main()
