"""
شبكة عصبية بسيطة (Value Network) بتاخد وضع الرقعة وتطلع رقم واحد
بين -1 و 1 يمثل: مين أقرب للفوز في الوضع ده من منظور اللاعب الأبيض.
  +1  => الأبيض في وضع فايز جدًا
  -1  => الأسود في وضع فايز جدًا
   0  => وضع متعادل تقريبًا

الشبكة دي هي "الدماغ" اللي بيتعلم من مباريات اللعب ضد النفس.
"""

import torch
import torch.nn as nn

from .encoding import NUM_PLANES


class ChessValueNet(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # يخرج قيمة بين -1 و 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.head(x).squeeze(-1)


def load_or_create_model(path: str, device: str = "cpu") -> ChessValueNet:
    """يحمّل موديل محفوظ لو موجود، أو ينشئ موديل جديد عشوائي."""
    import os

    model = ChessValueNet()
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"تم تحميل الموديل من {path}")
    else:
        print(f"لا يوجد موديل محفوظ في {path}، تم إنشاء موديل جديد عشوائي")
    model.to(device)
    return model
  
