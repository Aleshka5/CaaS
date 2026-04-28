import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self, input_height: int = 48, input_width: int = 112) -> None:
        super(CNNModel, self).__init__()

        # Общие параметры
        self.dropout_rate_1 = 0.3
        self.dropout_rate_2 = 0.5
        self.dropout_rate_3 = 0.2

        # Ветвь 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=0
            ),  # (48, 112) -> (24, 57)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate_1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (24, 57) -> (24, 57)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate_1),
            nn.Conv2d(
                in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=0
            ),  # (24, 57) -> (12, 28)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Ветвь 2 (идентична ветви 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=0
            ),  # (48, 112) -> (24, 57)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate_1),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # (24, 57) -> (24, 57)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate_1),
            nn.Conv2d(
                in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=0
            ),  # (24, 57) -> (12, 28)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Размер вектора после concat зависит от H×W входа (не только от высоты).
        with torch.no_grad():
            flattened_size = int(
                self.branch1(torch.zeros(1, 3, input_height, input_width)).numel() * 2
            )

        # Финальная часть
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(flattened_size),
            nn.Dropout(self.dropout_rate_2),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate_1),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_rate_3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        # Обработка двух входных тензоров
        x1 = self.branch1(img1)  # Ветвь 1
        x2 = self.branch2(img2)  # Ветвь 2

        # Flatten
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)

        # Объединение (concatenate)
        conc = torch.cat((x1, x2), dim=1)

        # Полносвязные слои
        out = self.fc_layers(conc)
        return out

    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
