import numpy as np


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        # Board is a 1D vector with 9 elements
        # 0 = blank , 1 = Player 1, -1 = Player 2
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.game_over = False
        return self.board

    def get_valid_moves(self):
        return np.where(self.board == 0)[0]

    def step(self, action):
        """
        Making a move
        Args:
            action (int): Field index (0-8)
        Returns:
            board: Board state after the move
            reward: 1 win, -1 loss, 0 game in progress)
            done: If the game ended
        """
        if self.board[action] != 0:
            raise ValueError(f"Move {action} is illegal!")

        self.board[action] = self.current_player

        # Chceck result
        if self.check_win():
            reward = 1
            self.game_over = True
        elif self.is_draw():
            reward = 0
            self.game_over = True
        else:
            reward = 0
            self.game_over = False
            self.current_player *= -1

        return self.board.copy(), reward, self.game_over

    def check_win(self):
        """Function to check if current player has won"""
        b = self.board.reshape(3, 3)
        player = self.current_player

        if np.any(np.sum(b, axis=0) == player * 3) or np.any(
            np.sum(b, axis=1) == player * 3
        ):
            return True

        # Diagonals
        if np.trace(b) == player * 3 or np.trace(np.fliplr(b)) == player * 3:
            return True

        return False

    def is_draw(self):
        return 0 not in self.board

    def __str__(self):
        """Simple board visualization"""
        symbols = {0: ".", 1: "X", -1: "O"}
        b = self.board.reshape(3, 3)
        out = ""
        for row in b:
            out += " ".join([symbols[x] for x in row]) + "\n"
        return out


if __name__ == "__main__":
    game = TicTacToe()
    game.step(0)  # X
    game.step(4)  # O
    game.step(1)  # X
    game.step(5)  # O
    game.step(8)  # X
    game.step(3)  # O win
    print(game)
    print("End of the game?", game.game_over)
