# opponent_cfr.py
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from cfr2 import game, cfr_player, random_player, solve, vs_random  # import base CFR components

# Data structures
# Counts: maps infoset string -> Counter(action_id -> count)
Counts = Dict[str, Counter]
# LegalByKey: maps infoset string -> tuple of legal actions (stable)
LegalByKey = Dict[str, Tuple[int, ...]]

def play_match_with_log(p0, p1) -> Tuple[np.ndarray, List[Tuple[str, int, List[int]]]]:
    """
    Like play_match, but also logs the opponent’s (infoset_key, action, legal_actions).
    Opponent = player 1.
    """
    opponent_obs: List[Tuple[str, int, List[int]]] = []
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            a = np.random.choice(actions, p=probs)
            state.apply_action(a)
        else:
            if state.current_player() == 0:
                a = p0(state)
            else:
                a = p1(state)
                key = state.information_state_string(1)   # infoset string for P1 (includes private card)
                legal_actions = state.legal_actions()     # list of action ids
                opponent_obs.append((key, a, legal_actions))
            state.apply_action(a)
    return np.array(state.rewards()), opponent_obs


def collect_opponent_counts(n_games: int, infoset = None) -> Tuple[Counts, LegalByKey]:
    """
    Play multiple games vs a random opponent and collect action counts for P1.
    Returns:
      - counts: mapping infoset -> Counter(action -> count)
      - legal_by_key: mapping infoset -> tuple of legal actions
    """
    counts: Counts = defaultdict(Counter)
    legal_by_key: LegalByKey = {}

    if infoset is None: infoset = {}

    p0 = cfr_player(infoset, player=0)  # our agent (CFR player)
    p1 = random_player()                   # opponent = random policy

    for _ in tqdm(range(n_games)):
        _, obs = play_match_with_log(p0, p1)
        for key, a, legal_actions in obs:
            # update count for this action
            counts[key][a] += 1

            # store legal actions (once per infoset)
            la = tuple(sorted(legal_actions))
            if key not in legal_by_key:
                legal_by_key[key] = la
            else:
                # optional: sanity check that legal actions are consistent
                assert legal_by_key[key] == la, f"Inconsistent legal actions for infoset {key}"

    return counts, legal_by_key


def estimate_policy_from_counts(
    counts: Counts,
    legal_by_key: LegalByKey,
    laplace: float = 1.0
) -> Dict[str, Dict[int, float]]:
    """
    Estimate the opponent policy π̂(a|I=key) from counts using
    Maximum Likelihood Estimation + Laplace smoothing.
    """
    pi_hat: Dict[str, Dict[int, float]] = {}
    for key, counter in counts.items():
        acts = legal_by_key[key]
        total = sum(counter[a] for a in acts)
        denom = total + laplace * len(acts)
        pi_hat[key] = {a: (counter[a] + laplace) / denom for a in acts}
    return pi_hat


def main():
    counts1, legal_by_key1 = collect_opponent_counts(n_games=50000)

    steps, vss, exploits, infoset = solve(t_max = 1000)

    os.makedirs("opponent_cfr", exist_ok=True)
    os.makedirs("opponent_cfr/cfr", exist_ok=True)

    plt.figure()
    plt.plot(steps, exploits)
    plt.xlabel("CFR Iterations")
    plt.ylabel("Exploitability")
    plt.savefig("opponent_cfr/cfr/exploit.png")

    plt.figure()
    plt.plot(steps, vss)
    plt.xlabel("CFR Iterations")
    plt.ylabel("Average Payoff vs Random")
    plt.savefig("opponent_cfr/cfr/vs_random.png")

    avg_payoff_rand = vs_random(infoset, n_games=10000)
    print(f"Average payoff vs random after CFR training: {avg_payoff_rand}")

    counts2, legal_by_key2 = collect_opponent_counts(n_games=50000, infoset=infoset)

    # Merge counts and legal actions from both phases
    counts = counts1
    for key, counter in counts2.items():
        counts[key].update(counter)
    legal_by_key = legal_by_key1
    for key, la in legal_by_key2.items():
        if key not in legal_by_key:
            legal_by_key[key] = la
        else:
            assert legal_by_key[key] == la, f"Inconsistent legal actions for infoset {key}"
    

    pi_hat = estimate_policy_from_counts(counts, legal_by_key)

    # Example: print one estimated policy
    some_key = next(iter(pi_hat))
    print("Example infoset:", some_key)
    print("Legal actions:", legal_by_key[some_key])
    print("Estimated π̂:", pi_hat[some_key])

    steps_rand, vss_rand, exploits_rand, infoset_rand = solve(t_max = 1000, mode="vs_fixed", pi_opp=pi_hat, opp_player = 1)

    os.makedirs("opponent_cfr/fixed_opp", exist_ok=True)

    plt.figure()
    plt.plot(steps_rand, exploits_rand)
    plt.xlabel("CFR Iterations")
    plt.ylabel("Exploitability")
    plt.savefig("opponent_cfr/fixed_opp/exploit.png")

    plt.figure()
    plt.plot(steps_rand, vss_rand)
    plt.xlabel("CFR Iterations")
    plt.ylabel("Average Payoff vs Random")
    plt.savefig("opponent_cfr/fixed_opp/vs_random.png")

    avg_payoff_rand = vs_random(infoset_rand, n_games=10000)
    print(f"Average payoff vs random after CFR training against fixed opponent: {avg_payoff_rand}")

if __name__ == "__main__":
    main()