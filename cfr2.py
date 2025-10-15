import random
import pyspiel
from pyspiel import State
import numpy as np
from dataclasses import dataclass
import random
from matplotlib import pyplot as plt
import tqdm
from open_spiel.python.algorithms import sequence_form_lp, exploitability
from open_spiel.python import policy
import pyspiel

game = pyspiel.load_game("kuhn_poker")


#(v1, v2, pi1, pi2) = sequence_form_lp.solve_zero_sum_game(game)
#merged_policy = policy.merge_tabular_policies([pi1, pi2], game)
#exploitability.exploitability(game, merged_policy)


#tabular_policy_player2 = policy.TabularPolicy(game)

class Info:

  def __init__(self, legal_actions: list[int]):

    self.n = len(legal_actions)
    self.legal_actions: np.ndarray = legal_actions
    self.cum_strategy: np.ndarray = np.zeros(self.n)
    self.cum_regret: np.ndarray = np.zeros(self.n)
    self.strategy: np.ndarray = np.ones(self.n) / self.n
    self.policy: np.ndarray = np.ones(self.n) / self.n

  def update_strategy_from_cum_regret(self):

    R_plus = np.maximum(self.cum_regret, 0.0)
    denominator = R_plus.sum()

    if denominator > 0:
      self.strategy = R_plus / denominator
    else:
      self.strategy = np.ones(self.n) / self.n

  def update_policy_from_cum_strategy(self):

      denominator = self.cum_strategy.sum()
      if denominator > 0:
        self.policy = self.cum_strategy / denominator
      else:
        self.policy = np.ones(self.n) / self.n

  @classmethod
  def get_info(cls, infoset: "Infoset", state: State, player: int) -> "Info":

    observation = state.observation_tensor(player)
    legal_actions = state.legal_actions()

    key = (state.information_state_string(player))

    if key not in infoset:
      infoset[key] = Info(legal_actions)

    info = infoset[key]
    assert info.legal_actions == legal_actions

    return info

Infoset = dict[tuple[str, int], Info]

def apply_action(state: State, action: int) -> State:
  next_state = state.clone()
  next_state.apply_action(action)
  return next_state


def random_player():

  def act(state: State):
    legal_actions = state.legal_actions()
    return random.choice(legal_actions)

  return act

def update_policy(infoset: Infoset):
  for info in infoset.values():
    info.update_policy_from_cum_strategy()

def update_strategy(infoset: Infoset):
  for info in infoset.values():
    info.update_strategy_from_cum_regret()

def cfr_player(infoset: Infoset, player: int=0):

  def act(state: State):
    info = Info.get_info(infoset, state, player)
    action = np.random.choice(info.legal_actions, p=info.policy)
    return action

  return act

def play_match(p0, p1) -> np.ndarray:

  state = game.new_initial_state()
  while not state.is_terminal():

    if state.is_chance_node():
      outcomes_with_probs = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes_with_probs)
      action = np.random.choice(action_list, p=prob_list)
      state.apply_action(action)
    else:

      if state.current_player() == 0:
        action = p0(state)
      else:
        action = p1(state)

      state.apply_action(action)

  return np.array(state.rewards())

def play_games(p0, p1, n_games: int = 50):

  rewards = np.zeros(2)
  for _ in range(n_games):
    rewards += play_match(p0, p1)
  return rewards / n_games

def vs_random(infoset: Infoset, n_games: int = 50) -> float:

  p1 = random_player()
  p0 = cfr_player(infoset)

  return play_games(p0, p1, n_games)[0]


def best_response(infoset: Infoset, state: State, br_player: int = 1):
  if state.is_terminal():
    return state.rewards()[br_player]


  if state.is_chance_node():
    expected_reward = 0

    for action, prob in state.chance_outcomes():
      next_state = apply_action(state, action)
      expected_reward += prob * best_response(infoset, next_state, br_player)

    return expected_reward

  current_player = state.current_player()
  info = Info.get_info(infoset, state, current_player)

  if current_player == br_player:
    values = []
    for action in state.legal_actions():
      next_state = apply_action(state, action)
      values.append(best_response(infoset, next_state, br_player))
    return max(values)

  else:
    value = 0
    for action, p_action in zip(info.legal_actions, info.policy, strict=True):
      next_state = apply_action(state, action)
      value += p_action * best_response(infoset, next_state, br_player)

    return value

def exploitability1(infoset: Infoset) -> float:
    state = game.new_initial_state()
    br0 = best_response(infoset, state, br_player=0)
    state = game.new_initial_state()
    br1 = best_response(infoset, state, br_player=1)
    return (1/2) * (br0 + br1)


# my, op, env
STRUE = True
weighting_specs = {
    "op_r_my_s": {
        "r": [False, STRUE, False],
        "s": [STRUE, False, False],
    },
    "op_env_r_my_s": {
        "r": [False, STRUE, STRUE],
        "s": [STRUE, False, False],
    },
    "my_r_op_s": {
        "r": [STRUE, False, False],
        "s": [False, STRUE, False],
    },
    "both_my": {
        "r": [STRUE, False, False],
        "s": [STRUE, False, False],
    },
    "both_my_env": {
        "r": [STRUE, False, STRUE],
        "s": [STRUE, False, STRUE],
    },
    "both_op_env": {
        "r": [False, STRUE, STRUE],
        "s": [False, STRUE, STRUE],
    },
    "both_op":  {
        "r": [False, STRUE, False],
        "s": [False, STRUE, False],
    },
    "full": {
        "r": [STRUE, STRUE, STRUE],
        "s": [STRUE, STRUE, STRUE],
    }
}

weighting: str | None = None



def cfr(infosets: Infoset, state: State, player_i: int, t: int, p0_p: float=1.0, p1_p: float=1.0, p_c: float=1.0):
    if state.is_terminal():
        return state.rewards()[player_i]


    if state.is_chance_node():
        outcomes = state.chance_outcomes() 
        value = 0.0
        for action, prob in outcomes:
            next_state = state.child(action)
            value += prob * cfr(infosets, next_state, player_i, t, p0_p, p1_p, p_c * prob)
        return value

    curr_player = state.current_player()

    infoset =  Info.get_info(infosets, state, curr_player)
  
    positive_regret = np.maximum(infoset.cum_regret, 0)
    if positive_regret.sum() > 0:
        infoset.strategy = positive_regret / positive_regret.sum()
    else:
        infoset.strategy = np.ones(infoset.n) / infoset.n

    node_value = 0.0
    action_values = np.zeros(infoset.n)

    for idx, a in enumerate(infoset.legal_actions):
        next_state = state.child(a)
        if curr_player == 0:
            v = cfr(infosets, next_state, player_i, t, p0_p * infoset.strategy[idx], p1_p, p_c)
        elif curr_player == 1:
            v = cfr(infosets, next_state, player_i, t, p0_p, p1_p * infoset.strategy[idx], p_c)
        else:
            v = cfr(infosets, next_state, player_i, t, p0_p, p1_p, p_c)
        action_values[idx] = v
        node_value += infoset.strategy[idx] * v

    if curr_player == player_i:
        opp_reach = p1_p if player_i == 0 else p0_p
        reach_weight = opp_reach * p_c
        infoset.cum_regret += reach_weight * (action_values - node_value)
        infoset.cum_strategy += (p0_p if player_i == 0 else p1_p) * infoset.strategy

    return node_value


    

def cfr_old(infoset: Infoset, state: State, player_i: int, t: int, p0_p: float = 1.0, p1_p: float = 1.0, env_p: float = 1.0):

  if state.is_terminal():
    return state.rewards()[player_i]

  if state.is_chance_node():
    expected_reward = 0

    for action, prob in state.chance_outcomes():
      next_state = apply_action(state, action)
      expected_reward += prob * cfr_old(infoset, next_state, player_i, t, p0_p, p1_p, env_p * prob)

    return expected_reward

  # Player node calc regret
  current_player = state.current_player()
  info = Info.get_info(infoset, state, current_player)

  expected_values_given_actions = np.zeros(info.n)
  expected_value_from_this_node = 0

  legal_actions = state.legal_actions()

  assert len(legal_actions) == info.n, "Mismatch in legal action size"
  assert sorted(legal_actions) == legal_actions, "Not sorted, it will break"


  for i_action, action in enumerate(legal_actions):
    p_action = info.strategy[i_action]
    next_state = apply_action(state, action)

    if current_player == 0:
      r = cfr_old(infoset, next_state, player_i, t, p0_p * p_action, p1_p, env_p)
    elif current_player == 1:
      r = cfr_old(infoset, next_state, player_i, t, p0_p, p1_p * p_action, env_p)
    else:
      assert False, f"Curr player = {current_player}"

    # From player i perspective
    expected_values_given_actions[i_action] = r
    expected_value_from_this_node += p_action * r

  p_i = p0_p if player_i == 0 else p1_p
  p_minus_i = p1_p if player_i == 0 else p0_p

  if current_player == 0:
    info.cum_strategy += (t + 1) * p0_p * info.strategy   # CFR+ linear weighting
  else:  # current_player == 1
    info.cum_strategy += (t + 1) * p1_p * info.strategy


  if current_player == player_i:

    # Update nodes from i perspective
    regret = (expected_values_given_actions - expected_value_from_this_node)
    info.cum_regret +=  env_p * p_minus_i * regret

    # Original placement
    # info.cum_strategy +=  p_i * info.strategy
    # CFR+
    # info.cum_strategy += t * p_i * info.strategy
    info.cum_regret = np.maximum(info.cum_regret, 0.0)

  return expected_value_from_this_node


def create_tabular_policy(infosets):
  tabular_policy= policy.TabularPolicy(game)
  for state_key, infoset in infosets.items():
    i = tabular_policy.state_lookup[state_key]
    policy_probs = np.zeros(game.num_distinct_actions())
    policy_probs[infoset.legal_actions] = infoset.policy
    tabular_policy.action_probability_array[i] = policy_probs


  return tabular_policy


def solve():

  exploits = []
  vss = []
  steps = []

  infoset: Infoset = dict()
  t_max = 200

  for t in tqdm.tqdm(list(range(t_max))):

    # Update strategies from cumulative regret for each infonode
    update_strategy(infoset)
    for i in range(2):
      state = game.new_initial_state()
      cfr_old(infoset, state, i, t)




    if (t % 50 == 0) or (t == (t_max - 1)):

      # Calculate the final policy from cumulative strategy for each infonode
      #avg_pi = compute_average_policy(infoset)
      update_policy(infoset)
      tabular_policy = create_tabular_policy(infoset)
      e = exploitability.exploitability(game, tabular_policy)


      #e = exploitability(infoset)
      #pyspiel.exploitability(game, avg_policy)
      vs = vs_random(infoset, n_games=10)

      steps.append(t)
      exploits.append(e)
      vss.append(vs)
  return steps, vss, exploits

steps, vss, exploits = solve()

plt.plot(steps, exploits)
plt.savefig("exploit.png")