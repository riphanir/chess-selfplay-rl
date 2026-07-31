"""
يدرّب الشبكة العصبية على بيانات مباريات اللعب ضد النفس اللي جهزها
self_play.py، ثم يحفظ نسخة محدثة من الموديل.

فكرة التدريب بسيطة (Monte-Carlo return): كل موقف مرّت بيه اللعبة
بياخد نفس نتيجة اللعبة النهائية كـ "الهدف" (target) اللي المفروض
الشبكة تتعلم تتوقعه.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from .model import load_or_create_model


def main():
    parser = argparse.ArgumentParser(description="تدريب الشبكة على بيانات اللعب ضد النفس")
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--data", default="data/latest_run.npz")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    model = load_or_create_model(args.model, device=device)
    model.train()

    data = np.load(args.data)
    states = torch.tensor(data["states"], dtype=torch.float32)
    targets = torch.tensor(data["targets"], dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(states, targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    final_loss = None
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch_states, batch_targets in loader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_states)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        final_loss = avg_loss
        print(f"  حقبة (epoch) {epoch + 1}/{args.epochs}: متوسط الخطأ = {avg_loss:.4f}")

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    torch.save(model.state_dict(), args.model)
    print(f"\nتم حفظ الموديل المحدث في {args.model}")

    # حفظ الخطأ النهائي عشان نضيفه لملف الإحصائيات
    with open("data/last_train_loss.txt", "w", encoding="utf-8") as f:
        f.write(f"{final_loss:.4f}\n")


if __name__ == "__main__":
    main()
