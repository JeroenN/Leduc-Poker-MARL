from collections import defaultdict, Counter
import numpy as np
import os
from open_spiel.python import policy
from utils import (
    save_comparison_with_ci,
    save_gap_mean_comparison,
    save_exploitability_comparison,
)
import argparse

from cfr import game, random_player, solve, vs_random, mixed_player, vs_mixed

# Maps infoset key -> counts of actions taken in that infoset
Counts = dict[str, Counter[int]]
# Maps infoset key -> stable tuple of legal actions for that infoset
LegalByKey = dict[str, tuple[int, ...]]


def play_match_with_log(
    p0, p1, log_seat: int
) -> tuple[np.ndarray, list[tuple[str, int, list[int]]]]:
    """
    Play one Leduc match between policies p0 (seat 0) and p1 (seat 1).

    We execute the game until terminal and record (infoset_key, action, legal_actions)
    only for the player sitting in `log_seat`.
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


def play_and_log(
    opponent_policy_fn, self_policy_fn, log_seat: int, n_games: int
) -> tuple[Counts, LegalByKey]:
    """
    Play multiple matches with fixed seating, logging the opponent's behaviour across episodes.
    """
    counts = defaultdict(Counter)
    legal_by_key = {}
    for _ in range(n_games):
        _, obs = play_match_with_log(
            self_policy_fn, opponent_policy_fn, log_seat=log_seat
        )
        for key, a, legal in obs:
            counts[key][a] += 1
            la = tuple(sorted(legal))
            legal_by_key.setdefault(key, la)
            assert legal_by_key[key] == la
    return counts, legal_by_key


def collect_opponent_counts(
    n_games: int, infoset=None, opponent_type: str = "random"
) -> tuple[Counts, LegalByKey]:
    """
    Collect empirical action frequencies for an opponent, from both seats.

    Our own agent always plays a random policy to ensure broad coverage
    of the game tree. The opponent policy can be either uniformly random
    or a 'mixed' policy (e.g. partially CFR-derived).
    """
    if infoset is None:
        infoset = {}

    # build opponent policies for both seats
    if opponent_type == "mixed":
        opp_as_p0 = mixed_player(infoset, player=0)
        opp_as_p1 = mixed_player(infoset, player=1)
    else:
        opp_as_p0 = random_player()
        opp_as_p1 = random_player()

    # our side always plays random for coverage
    self_as_p0 = random_player()
    self_as_p1 = random_player()

    # case A: opponent sits in seat 1, we log seat 1
    counts_A, legal_A = play_and_log(
        opponent_policy_fn=opp_as_p1,
        self_policy_fn=self_as_p0,
        log_seat=1,
        n_games=n_games,
    )

    # case B: opponent sits in seat 0, we log seat 0
    counts_B, legal_B = play_and_log(
        opponent_policy_fn=opp_as_p0,
        self_policy_fn=self_as_p1,
        log_seat=0,
        n_games=n_games,
    )

    # merge A and B
    counts = defaultdict(Counter)
    legal_by_key = {}

    for src_counts in (counts_A, counts_B):
        for key, counter in src_counts.items():
            counts[key].update(counter)

    for src_legal in (legal_A, legal_B):
        for key, la in src_legal.items():
            if key in legal_by_key:
                assert legal_by_key[key] == la
            else:
                legal_by_key[key] = la

    return counts, legal_by_key


def estimate_policy_from_counts(
    counts: Counts, legal_by_key: LegalByKey, laplace: float = 1.0
) -> dict[str, dict[int, float]]:
    """
    Convert empirical action counts into a stochastic opponent policy.

    For each infoset key, we take the observed action counts and apply
    Laplace-smoothed maximum-likelihood estimation to obtain a probability
    distribution over legal actions.
    """
    pi_hat: dict[str, dict[int, float]] = {}
    for key, counter in counts.items():
        acts = legal_by_key[key]
        total = sum(counter[a] for a in acts)
        denom = total + laplace * len(acts)
        pi_hat[key] = {a: (counter[a] + laplace) / denom for a in acts}
    return pi_hat


def dict_policy_to_tabular(game, pi_dict: dict[str, dict[int, float]]):
    """
    Convert a dict-based policy into an OpenSpiel TabularPolicy.
    """
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
    parser = argparse.ArgumentParser(
        description="Train CFR and OCFR against different opponent types."
    )
    parser.add_argument(
        "--opponent_type",
        type=str,
        default="random",
        choices=["random", "mixed"],
        help="Opponent policy class used for data collection and evaluation.",
    )
    args = parser.parse_args()

    OPPONENT_TYPE = args.opponent_type
    TRAIN_ITERS = 500
    N_GAMES_COLLECT = 100_000
    N_GAMES_EVAL = 50_000

    # CFR self-play training
    steps, vss, vss_se, exploits, gap_mean_hist, infoset = solve(t_max=TRAIN_ITERS)

    # collect opponent counts for OCFR
    counts, legal_by_key = collect_opponent_counts(
        n_games=N_GAMES_COLLECT, infoset=infoset, opponent_type=OPPONENT_TYPE
    )

    # eval CFR vs baseline opp
    if OPPONENT_TYPE == "mixed":
        avg_payoff_cfr = vs_mixed(infoset, n_games=N_GAMES_EVAL)
    else:
        avg_payoff_cfr = vs_random(infoset, n_games=N_GAMES_EVAL)
    print(f"[CFR] Average payoff vs {OPPONENT_TYPE}: {avg_payoff_cfr}")

    # estimate opponent policy from counts
    pi_hat = estimate_policy_from_counts(counts, legal_by_key)
    eval_opp_tab = dict_policy_to_tabular(game, pi_hat)

    # OCFR training (vs fixed opponent)
    (
        steps_fixed,
        vss_fixed,
        vss_fixed_se,
        exploits_fixed,
        gap_mean_hist_fixed,
        infoset_fixed,
    ) = solve(
        t_max=TRAIN_ITERS, mode="vs_fixed", pi_opp=pi_hat, eval_opp_tab=eval_opp_tab
    )

    os.makedirs("comparison", exist_ok=True)

    y_axis_avg_payoff_plot = (
        "Average Payoff vs Mixed"
        if OPPONENT_TYPE == "mixed"
        else "Average Payoff vs Random"
    )

    save_comparison_with_ci(
        steps,
        vss,
        np.array(vss_se),
        "CFR (self-play)",
        steps_fixed,
        vss_fixed,
        np.array(vss_fixed_se),
        "OCFR (vs fixed)",
        "CFR Iterations",
        y_axis_avg_payoff_plot,
        f"comparison/vs_{OPPONENT_TYPE}_comparison_std.pdf",
    )

    save_exploitability_comparison(
        steps,
        exploits,
        "CFR (self-play)",
        steps_fixed,
        exploits_fixed,
        "OCFR (vs fixed)",
        filename="comparison/exploitability_comparison.pdf",
    )

    save_gap_mean_comparison(
        steps,
        gap_mean_hist,
        steps_fixed,
        gap_mean_hist_fixed,
        filename="comparison/gap_to_br_mean.pdf",
    )

    # final eval OCFR
    if OPPONENT_TYPE == "mixed":
        avg_payoff_ocfr = vs_mixed(infoset_fixed, n_games=N_GAMES_EVAL)
    else:
        avg_payoff_ocfr = vs_random(infoset_fixed, n_games=N_GAMES_EVAL)
    print(f"[OCFR] Average payoff vs {OPPONENT_TYPE}: {avg_payoff_ocfr}")


if __name__ == "__main__":
    main()
