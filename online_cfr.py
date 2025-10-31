from copy import deepcopy
import pickle
from pathlib import Path
import pyspiel
import numpy as np
import tqdm
from pyspiel import State
from open_spiel.python.algorithms import best_response
from open_spiel.python import policy

from cfr import cfr, Infoset, solve, Info, Strategy, cfr_player, strategy_player, update_strategy, update_policy, apply_action

Counts = dict[str, dict[int, int]]

def save_nash(infoset: Infoset):
    
    with open(Path(__file__).parent  / "nash.pkl", "wb") as f:
        pickle.dump(infoset, f)

def load_nash() -> Infoset | None:

    path = Path(__file__).parent  / "nash.pkl"

    if not path.is_file():
        return None

    with open(path, "rb") as f:
        nash = pickle.load(f)
    
    return nash

def get_nash() -> Infoset:

    nash = load_nash()

    if nash is None:
        result = solve(t_max=500)
        nash = result[-1]
        save_nash(nash)

    return nash
    

def get_online_strategy_signal(steps_per_cycle: int) -> np.ndarray:

    n_cycles = 2
    steps = n_cycles * steps_per_cycle
    
    t = np.linspace(0, n_cycles, num=steps, endpoint=True)
    signal = (np.cos(2 * np.pi * 1.0 * t) + 1) / 2
    return signal

def mix_strategy(nash: Infoset, strategy_mixer: float) -> Strategy:

    def mix_node(node: Info) -> dict[int, float]:

        policy = node.policy
        uniform = np.ones(policy.size) / policy.size
        mixed = strategy_mixer * policy + (1 - strategy_mixer) * uniform

        return {
            action_int: action_p for action_int, action_p in zip(node.legal_actions, mixed, strict=True)
        }
    
    strategy = {key: mix_node(node) for key, node in nash.items()}
    return strategy
        

def get_first_estimate(nash: Infoset) -> Strategy:
    
    strategy = Strategy()

    for key, node in nash.items():

        snode = dict()
        
        for action_int, action_p in zip(node.legal_actions, node.policy, strict=True):
            snode[action_int] = action_p
        
        strategy[key] = snode
    
    return strategy


def update_op_policy_online(
    pi_hat: Strategy,
    counts: Counts,
    laplace: float,
) -> Strategy:
  

    updated_pi_hat = deepcopy(pi_hat) 
    for key, count in counts.items():

        action_dist = pi_hat[key]
        total = sum(count.values())

        if total == 0:
            continue

        # 1 => take incoming distribution completely
        update_factor = total / (total + laplace)
        
        # Calc the sampled action distribution
        incoming_dist = {action_int: (action_c / total) for action_int, action_c in count.items()}

        # Weighted avarage
        new_dist = {}
        for action_int, existing_p  in action_dist.items():

            incoming_p = incoming_dist.get(action_int, 0)
            new_dist[action_int] = (1 - update_factor)  * existing_p + update_factor * incoming_p

        updated_pi_hat[key] = new_dist
    
    return updated_pi_hat
                         

def play_match(p0, p1, log_seat: int, counts: Counts, game):
    state = game.new_initial_state()
    while not state.is_terminal():

        if state.is_chance_node():
            acts, probs = zip(*state.chance_outcomes())
            a = np.random.choice(acts, p=probs)
            state.apply_action(a)
        else:
            cur = state.current_player()
            a = p0(state) if cur == 0 else p1(state)
            if cur == log_seat:
                key = state.information_state_string(cur)
                
                count = counts.get(key, dict())
                c = count.get(a, 0)
                c += 1
                count[a] = c
                counts[key] = count

            state.apply_action(a)
    
    return state

def observe_op_play(p0, p1, log_seat: int, game, n_games: int = 10) -> Counts:
    counts = Counts()
    for _ in range(n_games):
        play_match(p0, p1, log_seat, counts, game)

    return counts

def strategy_to_tabular_policy(strategy: Strategy, game):
    tabular_policy = policy.TabularPolicy(game)
    for key, node in strategy.items():
        # if key not in tab.state_lookup:
        #     continue
        
        i = tabular_policy.state_lookup[key]
        probs = np.zeros(game.num_distinct_actions(), dtype=float)

        for action_int, action_p in node.items():
            probs[action_int] = action_p

        tabular_policy.action_probability_array[i] = probs
    return tabular_policy


def expected_reward_cfr_vs_strategy(
    infoset: Infoset,
    state: State,
    strategy: Strategy,
    learner: int,      
    p: float = 1.0
):

    if state.is_terminal():
        return state.rewards()[learner] * p

    if state.is_chance_node():
        expected_reward = 0
        for action, prob in state.chance_outcomes():
            next_state = apply_action(state, action)
            expected_reward += expected_reward_cfr_vs_strategy(infoset, next_state, strategy, learner, p * prob)

        return expected_reward
    
    legal_actions = state.legal_actions()
    if state.current_player() == learner:
        node = Info.get_info(infoset, state, learner)
        policy = {action: node.policy[i_action]  for i_action, action in enumerate(legal_actions)}
    else:
        key = state.information_state_string(1 - learner)
        policy = strategy[key]

    expected_reward = 0
    for action_int, action_p in policy.items():
        next_state = apply_action(state, action_int)
        expected_reward += expected_reward_cfr_vs_strategy(infoset, next_state, strategy, learner, p * action_p)

    return expected_reward


def online_solve(
    steps_per_cycle: int,
    laplace: float,
    learner: int = 0,
):
    
    game = pyspiel.load_game("leduc_poker")

    opponent = 1 - learner
    infoset = get_nash()
    strategy_mixer_signal = get_online_strategy_signal(steps_per_cycle)
    op_strategy_estimate = get_first_estimate(infoset)


    if learner == 0:
        p0 = cfr_player(infoset, player=0)
        p0_fixed_strategy = None
    else:
        p1 = cfr_player(infoset, player=1)
        p1_fixed_strategy = None

    gaps_to_best_response = []
    for strategy_mixer in tqdm.tqdm(strategy_mixer_signal, disable=True):

        # Update internal CFR strategy
        update_strategy(infoset)

        # Form the actual op strategy
        op_strategy_actual = mix_strategy(infoset, strategy_mixer=strategy_mixer)
        if opponent == 0:
            p0 = strategy_player(op_strategy_actual, player=0)
        else:
            p1 = strategy_player(op_strategy_actual, player=1)

        # Play a game
        counts = observe_op_play(p0, p1, log_seat=opponent, game=game)

        # Update op strategy estimate
        op_strategy_estimate = update_op_policy_online(op_strategy_estimate, counts, laplace=laplace)
        if opponent == 0:
            p0_fixed_strategy = op_strategy_estimate
        else:
            p1_fixed_strategy = op_strategy_estimate

        # Learn BR to the op
        state = game.new_initial_state()

        for _ in range(20):
            cfr(
                infoset,
                state,
                player_i=learner,
                t=1_000,
                p0_fixed_strategy=p0_fixed_strategy,
                p1_fixed_strategy=p1_fixed_strategy
            )

        # Update metrics and my policy
        update_policy(infoset)

        learner_expected_reward = expected_reward_cfr_vs_strategy(
            infoset,
            state=game.new_initial_state(),
            strategy=op_strategy_actual,
            learner=learner
        )
        op_tabular_policy = strategy_to_tabular_policy(op_strategy_actual, game)


        # Should learner be there? Or should I pass opponent?
        br_to_op_value = best_response.BestResponsePolicy(game, learner, op_tabular_policy).value(game.new_initial_state())
        gaps_to_best_response.append(br_to_op_value - learner_expected_reward)

    error = np.mean(gaps_to_best_response) 
    
    return strategy_mixer_signal, gaps_to_best_response, error


import multiprocessing as mp
import csv
import random
import time
from pathlib import Path
import numpy as np

RESULTS_PATH = Path(__file__).parent / "grid_results.csv"


def random_logspace(low, high, num_samples):
    log_low, log_high = np.log10(low), np.log10(high)
    return np.power(10, np.random.uniform(log_low, log_high, num_samples))


def _init_result_file():
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["steps_per_cycle", "laplace", "error", "exception"])


def random_grid_search(n_samples: int, n_cpus: int):

    n_cpus = n_cpus or mp.cpu_count()
    _init_result_file()

    steps_per_cycle_samples = random_logspace(100, 100_000, n_samples).astype(int)
    #steps_per_cycle_samples = random_logspace(2, 10, n_samples).astype(int)
    laplace_samples = random_logspace(0.5, 50, n_samples)

    param_grid = list(zip(steps_per_cycle_samples, laplace_samples))

    with mp.Pool(processes=n_cpus, maxtasksperchild=1) as pool:
        pool.map(_grid_search_worker, param_grid)


def _write_result_safe(row: list):
    for _ in range(10):
        try:
            with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return
        except OSError:
            time.sleep(random.uniform(0.05, 0.2))
    print("Warning: Failed to write result after multiple retries:", row)


def _grid_search_worker(params):
    steps_per_cycle, laplace = params
    try:
        _, _, error = online_solve(
            steps_per_cycle=int(steps_per_cycle),
            laplace=float(laplace),
            learner=0,
        )
        row = [int(steps_per_cycle), float(laplace), float(error), ""]
        _write_result_safe(row)
    except Exception as e:
        row = [int(steps_per_cycle), float(laplace), "", str(e)]
        _write_result_safe(row)


if __name__ == "__main__":
    random_grid_search(n_samples=3, n_cpus=3)



# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     strategy_mixer_signal, gaps_to_best_response = online_solve(
#         steps_per_cycle=1000,
#         laplace=1.0,
#         learner=0,
#     )

#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()

#     l1, = ax1.plot(strategy_mixer_signal, color='tab:blue', label='strategy_mixer')
#     l2, = ax2.plot(gaps_to_best_response, color='tab:orange', label='gap_to_best_response')

#     ax1.legend(handles=[l1, l2], loc='upper right')
#     fig.savefig("./figures/online_cfr.png")
        
    