import random
from board import TicTacToe
from agent import Agent


def train():
    game = TicTacToe()
    agent = Agent()

    n_games = 0
    agent.epsilon = 1.0

    print("Training starts")

    while True:
        game.reset()
        done = False
        last_state_action = {1: None, -1: None}

        while not done:
            player = game.current_player
            valid_moves = game.get_valid_moves()

            # Model wants to see his moves as 1 not -1
            canonical_state = agent.get_state(game).copy() * player

            action = agent.get_action(canonical_state, valid_moves)

            _, reward, done = game.step(action)

            last_state_action[player] = (canonical_state, action)

            if done:
                if reward == 1:
                    # player won
                    next_state_winner = agent.get_state(game).copy() * player
                    w_state, w_action = last_state_action[player]
                    agent.remember(w_state, w_action, 1, next_state_winner, True)

                    # other player lost
                    other_player = player * -1
                    if last_state_action[other_player] is not None:
                        next_state_loser = agent.get_state(game).copy() * other_player
                        l_state, l_action = last_state_action[other_player]
                        agent.remember(l_state, l_action, -1, next_state_loser, True)

                elif reward == 0:
                    # Draw - both players get reward = 0
                    for p in [-1, 1]:
                        if last_state_action[p] is not None:
                            p_next_state = agent.get_state(game).copy() * p
                            p_state, p_action = last_state_action[p]
                            agent.remember(p_state, p_action, 0, p_next_state, True)

            else:
                # game still running
                # saving state for other player
                other_player = player * -1
                if last_state_action[other_player] is not None:
                    o_next_state = agent.get_state(game).copy() * other_player
                    o_state, o_action = last_state_action[other_player]
                    agent.remember(o_state, o_action, 0, o_next_state, False)

        # ---------- Learning after the game

        n_games += 1
        if len(agent.memory) > agent.batch_size:
            minibatch = random.sample(agent.memory, agent.batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            agent.train_step(states, actions, rewards, next_states, dones)

        if agent.epsilon > 0.05:
            agent.epsilon = max(0.05, agent.epsilon - 0.001)

        if n_games % 100 == 0:
            print(
                f"Game: {n_games} | Epsilon: {agent.epsilon:.2f} | Memory: {len(agent.memory)}"
            )
            if n_games % 1000 == 0:
                agent.model.save("models/tictactoe_net.pth")


if __name__ == "__main__":
    train()
