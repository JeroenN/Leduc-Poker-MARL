from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from cfr2 import game, cfr_player, random_player, solve, vs_random  # import base CFR components

# Counts: maps infoset string -> Counter(action_id -> count)
Counts = Dict[str, Counter]
# LegalByKey: maps infoset string -> tuple of legal actions (stable)
LegalByKey = Dict[str, Tuple[int, ...]]

def play_match_with_log(p0, p1, log_seat: int):
    """
    Play a match between p0 (seat 0) and p1 (seat 1).
    Log (infoset_key, action, legal_actions) ONLY for the player at `log_seat`.
    """
    obs = []
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
                obs.append((key, a, list(state.legal_actions())))
            state.apply_action(a)
    return np.array(state.rewards()), obs


def collect_opponent_counts(n_games: int, infoset = None) -> Tuple[Counts, LegalByKey]:
    """
    Collect action counts for the OPPONENT in both seating positions:
      - Case 1: You are seat 0, opponent is seat 1 -> log seat 1
      - Case 2: You are seat 1, opponent is seat 0 -> log seat 0
    """
    counts: Counts = defaultdict(Counter)
    legal_by_key: LegalByKey = {}

    if infoset is None: infoset = {}

    # --- Case 1: Opponent sits at seat 1 ---
    p_you_0 = cfr_player(infoset, player=0)
    p_opp_1 = random_player()
    for _ in range(n_games):
        _, obs = play_match_with_log(p_you_0, p_opp_1, log_seat=1)
        for key, a, legal in obs:
            counts[key][a] += 1
            la = tuple(sorted(legal))
            legal_by_key.setdefault(key, la)
            assert legal_by_key[key] == la

    # --- Case 2: Opponent sits at seat 0 ---
    p_opp_0 = random_player()
    p_you_1 = cfr_player(infoset, player=1)
    for _ in range(n_games):
        _, obs = play_match_with_log(p_opp_0, p_you_1, log_seat=0)
        for key, a, legal in obs:
            counts[key][a] += 1
            la = tuple(sorted(legal))
            legal_by_key.setdefault(key, la)
            assert legal_by_key[key] == la

    return counts, legal_by_key

def save_plot(x, y, xlabel, ylabel, filename):
    """Generate and save a plot."""
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


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

    os.makedirs("opponent_cfr/cfr", exist_ok=True)

    save_plot(steps, exploits, "CFR Iterations", "Exploitability",
            "opponent_cfr/cfr/exploit.png")
    save_plot(steps, vss, "CFR Iterations", "Average Payoff vs Random",
            "opponent_cfr/cfr/vs_random.png")


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
    

    # Estimate opponent policy π̂ from aggregated counts
    pi_hat = estimate_policy_from_counts(counts, legal_by_key)

    # Example: print one estimated policy
    some_key = next(iter(pi_hat))
    print("Example infoset:", some_key)
    print("Legal actions:", legal_by_key[some_key])
    print("Estimated π̂:", pi_hat[some_key])

    steps_rand, vss_rand, exploits_rand, infoset_rand = solve(t_max = 1000, mode="vs_fixed", pi_opp=pi_hat)

    os.makedirs("opponent_cfr/fixed_opp", exist_ok=True)

    save_plot(steps_rand, exploits_rand, "CFR Iterations", "Exploitability",
              "opponent_cfr/fixed_opp/exploit.png")
    save_plot(steps_rand, vss_rand, "CFR Iterations", "Average Payoff vs Random",
              "opponent_cfr/fixed_opp/vs_random.png")

    avg_payoff_rand = vs_random(infoset_rand, n_games=10000)
    print(f"Average payoff vs random after CFR training against fixed opponent: {avg_payoff_rand}")

if __name__ == "__main__":
    main()