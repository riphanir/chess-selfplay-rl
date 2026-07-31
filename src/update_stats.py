"""يجمع نتائج آخر دورة (لعب ضد النفس + تدريب + تقييم) في سطر واحد في stats.csv"""

import csv
import os
from datetime import datetime, timezone

STATS_FILE = "stats.csv"
HEADER = [
    "timestamp_utc",
    "games_played",
    "wins_white",
    "wins_black",
    "draws",
    "avg_game_length",
    "train_loss",
    "winrate_vs_baseline_pct",
]


def read_line(path, default=""):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return default


def main():
    summary = read_line("data/last_selfplay_summary.txt")
    games, wins_w, wins_b, draws, avg_len = (
        summary.split(",") if summary else ("", "", "", "", "")
    )
    loss = read_line("data/last_train_loss.txt")
    winrate = read_line("data/last_eval_winrate.txt")

    row = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        games,
        wins_w,
        wins_b,
        draws,
        avg_len,
        loss,
        winrate,
    ]

    file_exists = os.path.exists(STATS_FILE)
    with open(STATS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(HEADER)
        writer.writerow(row)

    print(f"تم تحديث {STATS_FILE}:")
    print(dict(zip(HEADER, row)))


if __name__ == "__main__":
    main()
