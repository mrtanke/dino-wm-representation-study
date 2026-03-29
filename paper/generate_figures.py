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
    plan_rows = df[df["experiment_id"].str.match(r"F\\d+_plan$")].copy()
    train_rows = df[df["experiment_id"].str.match(r"F\\d+_train$")].copy()

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

    axes[1].bar(train_loss.index, train_loss.values, color=colors)
    axes[1].set_title("Formal Matrix Mean Training Loss")
    axes[1].set_ylabel("Mean train loss")
    axes[1].tick_params(axis="x", rotation=20)

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
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "patch_large_eval.png", dpi=220, bbox_inches="tight")
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRACKER)
    save_formal_matrix(df)
    save_patch_large_eval(df)
    save_cross_env(df)


if __name__ == "__main__":
    main()
