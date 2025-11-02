# Opponent Counterfactual Regret (OCFR)

This project implements Opponent Counterfactual Regret (OCFR), an extension of Counterfactual Regret Minimization (CFR) for imperfect-information games. OCFR models the opponent’s behavior from observed play and learns a best response to that model, aiming for exploitation rather than equilibrium.

Using Leduc Hold’em in OpenSpiel, we compare OCFR and CFR on exploitability, best-response gap, and payoff.
We also test an online adaptation experiment, where OCFR updates its opponent model as the opponent’s strategy changes.
OCFR yields higher payoffs against fixed, suboptimal opponents but struggles when the opponent adapts rapidly.
