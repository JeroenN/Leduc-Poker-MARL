from collections import defaultdict, Counter
from typing import Dict, Tuple
import numpy as np
import os
from open_spiel.python import policy
from utils import save_comparison_with_ci, save_gap_mean_comparison, save_exploitability_comparison

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


def dict_policy_to_tabular(game, pi_dict):
    tab = policy.TabularPolicy(game)
    for key, table in pi_dict.items():
        if key not in tab.state_lookup:
            continue
        i = tab.state_lookup[key]
        probs = np.zeros(game.num_distinct_actions(), dtype=float)
        s = sum(table.values())
        if s <= 0:
            continue
        for a, p in table.items():
            probs[a] = p / s
        tab.action_probability_array[i] = probs
    return tab
    

def main():
    counts, legal_by_key = collect_opponent_counts(n_games=100000)

    steps, vss, vss_se, exploits, gap_mean_hist, infoset = solve(t_max = 500)

    avg_payoff_rand = vs_random(infoset, n_games=50000)
    print(f"Average payoff vs random after CFR training: {avg_payoff_rand}")

    # Estimate opponent policy π̂ from aggregated counts
    pi_hat = estimate_policy_from_counts(counts, legal_by_key)

    # Example: print one estimated policy
    some_key = next(iter(pi_hat))
    print("Example infoset:", some_key)
    print("Legal actions:", legal_by_key[some_key])
    print("Estimated π̂:", pi_hat[some_key])

    eval_opp_tab = dict_policy_to_tabular(game, pi_hat)

    steps_rand, vss_rand, vss_rand_se, exploits_rand, gap_mean_hist_rand, infoset_rand = solve(t_max = 500, mode="vs_fixed", pi_opp=pi_hat,eval_opp_tab=eval_opp_tab)
    
    os.makedirs("comparison", exist_ok=True)

    save_comparison_with_ci(
        steps, vss, np.array(vss_se), "CFR (self-play)",
        steps_rand, vss_rand, np.array(vss_rand_se), "OCFR (vs fixed)",
        "CFR Iterations", "Average Payoff vs Random",
        "comparison/vs_random_comparison_std.pdf"
    )

    save_exploitability_comparison(
        steps, exploits, "CFR (self-play)",
        steps_rand, exploits_rand, "OCFR (vs fixed)",
        filename="comparison/exploitability_comparison.pdf"
    )

    save_gap_mean_comparison(
        steps, gap_mean_hist,
        steps_rand, gap_mean_hist_rand,
        filename="comparison/gap_to_br_mean.pdf"
    )

    avg_payoff_rand = vs_random(infoset_rand, n_games=50000)
    print(f"Average payoff vs random after CFR training against fixed opponent: {avg_payoff_rand}")

if __name__ == "__main__":
    main()