import torch
from board import TicTacToe
from agent import Agent


def play():
    game = TicTacToe()
    agent = Agent()

    agent.model.load("models/tictactoe_net.pth")
    agent.model.eval()
    agent.epsilon = 0

    print("Start of the game!")
    user_starts = input("Wanna be X (1) or O (2)? [1/2]: ") == "1"

    # Setting roles
    user_player = 1 if user_starts else -1
    ai_player = user_player * -1

    while not game.game_over:
        print("\n" + str(game))
        valid_moves = game.get_valid_moves()

        if game.current_player == user_player:
            # Player's turn
            print(f"Legal moves: {valid_moves}")
            try:
                move = int(input("Choose your move (0-8): "))
                if move not in valid_moves:
                    print("Illegal move.")
                    continue
            except ValueError:
                print("Enter valid number!")
                continue
        else:
            # AIs' turn
            print("AI's turn...")
            canonical_state = agent.get_state(game).copy() * ai_player

            move = agent.get_action(canonical_state, valid_moves)
            print(f"AI's selected move: {move}")

        # Making real move
        game.step(move)

    # End of the game
    print("\n" + str(game))
    if game.check_win():
        if game.current_player == user_player:
            print("You won!")
        else:
            print("AI won!")
    else:
        print("Its a draw!")


if __name__ == "__main__":
    play()
