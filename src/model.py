import torch
import torch.nn as nn
import os


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer definition:
        # Input: 9 neurons (board state)
        # Hidden layers: 32 neurons
        # Output: 9 neurons (Q value for each field on the board)
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.net(x)

    def save(self, path="models/tictactoe_net.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="models/tictactoe_net.pth"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path} - starting from scratch")


if __name__ == "__main__":
    model = TicTacToeNet()
    print("Model architecture:")
    print(model)

    test_input = torch.tensor([0, 0, 1, -1, 1, -1, 0, 0, 0], dtype=torch.float32)

    output = model(test_input)

    print("\nQ-values for each field):")
    print(output.detach().numpy())

    model.save()
