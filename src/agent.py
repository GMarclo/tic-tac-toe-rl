import torch
import random
import numpy as np
from collections import deque
from model import TicTacToeNet

MAX_MEMORY = 100_000  # How many previous games model remember
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate

        self.memory = deque(maxlen=MAX_MEMORY)
        self.batch_size = BATCH_SIZE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TicTacToeNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = torch.nn.MSELoss()

    def get_state(self, game):
        return np.array(game.board, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, valid_moves):
        """
        Next move choice:
        1. Random - if rand_num < epsilon
        2. From Network - if rand_num >= epsilon
        """

        # Exploarion (random move):
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_moves)

        # Exploitation (network move):
        state0 = torch.tensor(state, dtype=torch.float).to(self.device)
        prediction = self.model(state0)

        move_mask = torch.ones(9).to(self.device) * -float("inf")
        move_mask[valid_moves] = 0

        masked_prediction = prediction + move_mask

        final_move = torch.argmax(masked_prediction).item()
        return final_move

    def train_step(self, state, action, reward, next_state, done):
        """
        Deep Q-Learning function
        Method can use as input single step or whole batch
        """
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(
            self.device
        )
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        # Tensor reshaping (n, ) -> (n,1)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            action = torch.unsqueeze(action, dim=0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new += self.gamma * torch.max(
                    self.model(next_state[idx])
                )  # Bellman's eq

            target[idx][action[idx].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
