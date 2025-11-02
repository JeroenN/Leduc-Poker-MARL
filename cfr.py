import random
import pyspiel
from pyspiel import State
import numpy as np
import random
import tqdm
from open_spiel.python.algorithms import exploitability, best_response
from open_spiel.python import policy
import pyspiel

game = pyspiel.load_game("leduc_poker")
Strategy = dict[str, dict[int, float]]


# ------------------Setup infosets and game stuff---------------------
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

    legal_actions = state.legal_actions()

    key = (state.information_state_string(player))

    if key not in infoset:
      infoset[key] = Info(legal_actions)

    info = infoset[key]
    assert info.legal_actions == legal_actions

    return info
  
  def equal_nodes_policies(self, other: "Info") -> bool:
     
    eq_la = np.array_equal(self.legal_actions, other.legal_actions)
    
    if not eq_la:
       return False
    
    return np.allclose(self.policy, other.policy, atol=1e-8)
  
  @staticmethod
  def compare_infosets(i1: "Infoset", i2: "Infoset") -> bool:
    
    ke = set(i1.keys()) == set(i2.keys())
    if not ke:
       return False
    
    for key in i1.keys():
      eq = i1[key].equal_nodes_policies(i2[key])

      if not eq:
         return False
      
    return True
    

Infoset = dict[str, Info]

def apply_action(state: State, action: int) -> State:
  next_state = state.clone()
  next_state.apply_action(action)
  return next_state


# ---------------------------- CFR code --------------------------------

def update_policy(infoset: Infoset):
  for info in infoset.values():
    info.update_policy_from_cum_strategy()

def update_strategy(infoset: Infoset):
  for info in infoset.values():
    info.update_strategy_from_cum_regret()
    

def cfr(
    infoset: Infoset,
    state: State,
    player_i: int,
    t: int,
    p0_p: float = 1.0,
    p1_p: float = 1.0,
    env_p: float = 1.0,
    p0_fixed_strategy: Strategy | None = None,
    p1_fixed_strategy: Strategy | None = None,
  ):

  if state.is_terminal():
    return state.rewards()[player_i]

  if state.is_chance_node():
    expected_reward = 0

    for action, prob in state.chance_outcomes():
      next_state = apply_action(state, action)
      expected_reward += prob * cfr(infoset, next_state, player_i, t, p0_p, p1_p, env_p * prob, p0_fixed_strategy, p1_fixed_strategy)

    return expected_reward

  # Player node calc regret
  current_player = state.current_player()
  current_fixed_strategy = (p0_fixed_strategy) if current_player == 0 else (p1_fixed_strategy)
  current_fixed = current_fixed_strategy is not None
  legal_actions = state.legal_actions()

  info = Info.get_info(infoset, state, current_player)
  assert len(legal_actions) == info.n, "Mismatch in legal action size"

  if current_fixed:
    
    key = state.information_state_string(current_player)
    fixed_node_strategy = current_fixed_strategy.get(key, None)

    if fixed_node_strategy is None:
      total_legal = len(legal_actions)
      fixed_node_strategy = {action_int: 1.0 / total_legal for action_int in legal_actions}


  expected_values_given_actions = np.zeros(info.n)
  expected_value_from_this_node = 0
  for i_action, action in enumerate(legal_actions):

    if current_fixed:
      # fixed_node_strategy is dict
      p_action = fixed_node_strategy[action]
    else:
      # info.strategy is array 
      p_action = info.strategy[i_action]

    next_state = apply_action(state, action)

    if current_player == 0:
      r = cfr(infoset, next_state, player_i, t, p0_p * p_action, p1_p, env_p, p0_fixed_strategy, p1_fixed_strategy)
    elif current_player == 1:
      r = cfr(infoset, next_state, player_i, t, p0_p, p1_p * p_action, env_p, p0_fixed_strategy, p1_fixed_strategy)
    else:
      assert False, f"Curr player = {current_player}"

    # From player i perspective
    expected_values_given_actions[i_action] = r
    expected_value_from_this_node += p_action * r

  p_i = p0_p if player_i == 0 else p1_p
  p_minus_i = p1_p if player_i == 0 else p0_p

  # This was working, come back here if broken, don't forget not to update for fixed player
  # info.cum_strategy += (t + 1) * p_i * info.strategy

  if current_player == player_i:

    # Update nodes from i perspective
    regret = (expected_values_given_actions - expected_value_from_this_node)
    info.cum_regret +=  env_p * p_minus_i * regret

    # Original placement
    # info.cum_strategy +=  p_i * info.strategy
    # CFR+
    # Will this work? 
    info.cum_strategy += (t + 1) * p_i * info.strategy
    info.cum_regret = np.maximum(info.cum_regret, 0.0)

  return expected_value_from_this_node

def cfr_vs_fixed_opponent(
    infosets: Infoset,
    state: State,
    learner: int,           # e.g., 0
    t: int,                 # iteration index (for averaging weights if you like)
    pi_opp: Strategy,  # learned opponent policy: key -> {action: prob}
    p0: float = 1.0,        # reach prob for P0 up to this node
    p1: float = 1.0,        # reach prob for P1 up to this node
    pc: float = 1.0         # reach prob for chance up to this node
):
    # terminal
    if state.is_terminal():
        return state.rewards()[learner]

    # chance
    if state.is_chance_node():
        v = 0.0
        for a, prob in state.chance_outcomes():
            child = state.child(a)
            v += prob * cfr_vs_fixed_opponent(infosets, child, learner, t, pi_opp, p0, p1, pc * prob)
        return v

    cur = state.current_player()

    # ---------------------------
    # Opponent node (fixed policy)
    # ---------------------------
    if cur != learner:
        key = state.information_state_string(cur)  # opponent infoset key
        acts = state.legal_actions()

        # Fetch π̂_opp(a|key); fallback to uniform over legal acts if missing
        probs = []
        table = pi_opp.get(key, None)
        if table is None:
            probs = np.ones(len(acts)) / len(acts)
        else:
            # collect in action order; if an action missing in dict, treat as 0 then renormalize
            vec = np.array([table.get(a, 0.0) for a in acts], dtype=float)
            s = vec.sum()
            if s <= 0:
                vec = np.ones(len(acts)) / len(acts)
            else:
                vec = vec / s
            probs = vec

        # Expected value under fixed opponent strategy
        node_v = 0.0
        for p_a, a in zip(probs, acts):
            child = state.child(a)
            if cur == 0:
                node_v += p_a * cfr_vs_fixed_opponent(infosets, child, learner, t, pi_opp, p0 * p_a, p1, pc)
            else:  # cur == 1
                node_v += p_a * cfr_vs_fixed_opponent(infosets, child, learner, t, pi_opp, p0, p1 * p_a, pc)
        return float(node_v)

    # ---------------------------
    # Learner node (regret-matching)
    # ---------------------------
    info = Info.get_info(infosets, state, cur)  # this creates/stores only learner’s infosets
    acts = info.legal_actions

    # current strategy from positive regrets (CFR/CFR+)
    R_plus = np.maximum(info.cum_regret, 0.0)
    if R_plus.sum() > 0:
        info.strategy = R_plus / R_plus.sum()
    else:
        info.strategy = np.ones(info.n) / info.n

    # recurse on actions
    util_a = np.zeros(info.n, dtype=float)
    node_v = 0.0
    for k, a in enumerate(acts):
        p_a = info.strategy[k]
        child = state.child(a)
        if cur == 0:
            util_a[k] = cfr_vs_fixed_opponent(infosets, child, learner, t, pi_opp, p0 * p_a, p1, pc)
        else:
            util_a[k] = cfr_vs_fixed_opponent(infosets, child, learner, t, pi_opp, p0, p1 * p_a, pc)
        node_v += p_a * util_a[k]

    # regret update (only for learner)
    opp_reach = p1 if learner == 0 else p0
    info.cum_regret += pc * opp_reach * (util_a - node_v)
    info.cum_regret = np.maximum(info.cum_regret, 0.0)  # CFR+ clamp (optional)

    # average strategy update for learner (you can use linear weighting t if desired)
    owner_reach = p0 if learner == 0 else p1
    info.cum_strategy += owner_reach * info.strategy  # or (t+1)*owner_reach for CFR+

    return float(node_v)

# ---------------------------- Evaluation code --------------------------------

def random_player():

  def act(state: State):
    legal_actions = state.legal_actions()
    return random.choice(legal_actions)

  return act

def mixed_player(infoset: Infoset, player: int=0):
    def act(state: State):
      info = Info.get_info(infoset, state, player)
      if random.random() < 0.5:
          action = np.random.choice(info.legal_actions, p=info.policy)
      else:
          action = random.choice(info.legal_actions)
      return action
    
    return act

def cfr_player(infoset: Infoset, player: int=0):

  def act(state: State):
    info = Info.get_info(infoset, state, player)
    action = np.random.choice(info.legal_actions, p=info.policy)
    return action

  return act

def strategy_player(strategy: Strategy, player: int=0):
   
  def act(state: State):
     
    key = state.information_state_string(player)
    node_strategy = strategy.get(key, None)

    if node_strategy is None:
      legal_actions = state.legal_actions()
      total_legal = len(legal_actions)
      node_strategy = {action_int: 1.0 / total_legal for action_int in legal_actions}
    
    actions = list(node_strategy.keys())
    p = list(node_strategy.values())
    
    return np.random.choice(actions, p=p)
  
  return act

def play_match(p0, p1) -> np.ndarray:

  opponent_obs: list[tuple[str, int]] = []

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
        key = state.information_state_string(1)  # infoset with opponent card
        opponent_obs.append((key, action))

      state.apply_action(action)

  return np.array(state.rewards())


def play_games(p0, p1, n_games: int = 50):
    """Play n_games and return  all rewards."""
    all_rewards = np.zeros((n_games, 2))
    for i in range(n_games):
        all_rewards[i] = play_match(p0, p1)
    return all_rewards


def vs_random(infoset: Infoset, n_games: int = 500):
    """Return mean and standard error of payoff vs random."""
    # Player 0 vs random
    rewards1 = play_games(cfr_player(infoset, 0), random_player(), n_games)
    # Random vs Player 1
    rewards2 = play_games(random_player(), cfr_player(infoset, 1), n_games)
    
    # Payoffs for the learner in both seatings
    payoffs = np.concatenate([rewards1[:,0], rewards2[:,1]])
    
    mean = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)
    se = std / np.sqrt(len(payoffs))
    return mean, se

def vs_mixed(infoset: Infoset, n_games: int = 500):
    """Return mean and standard error of payoff vs (50/50 CFR/random) player."""
    # Player 0 vs random
    rewards1 = play_games(cfr_player(infoset, 0), mixed_player(infoset, 1), n_games)
    # Random vs Player 1
    rewards2 = play_games(mixed_player(infoset, 0), cfr_player(infoset, 1), n_games)
    
    # Payoffs for the learner in both seatings
    payoffs = np.concatenate([rewards1[:,0], rewards2[:,1]])
    
    mean = np.mean(payoffs)
    std = np.std(payoffs, ddof=1)
    se = std / np.sqrt(len(payoffs))
    return mean, se


def create_tabular_policy(infosets):
  tabular_policy= policy.TabularPolicy(game)
  for state_key, infoset in infosets.items():
    i = tabular_policy.state_lookup[state_key]
    policy_probs = np.zeros(game.num_distinct_actions())
    policy_probs[infoset.legal_actions] = infoset.policy
    tabular_policy.action_probability_array[i] = policy_probs
  return tabular_policy


def value_vs_fixed_policy(game, learner_tab, opp_tab, learner_id, n_games=20000):
    def player_from_tab(tab_pol, pid):
        def act(state):
            ap = tab_pol.action_probabilities(state, pid)
            acts, probs = zip(*ap.items())
            return np.random.choice(acts, p=probs)
        return act

    payoffs = []
    for _ in range(n_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                acts, probs = zip(*state.chance_outcomes())
                state.apply_action(np.random.choice(acts, p=probs))
            else:
                cur = state.current_player()
                if cur == 0:
                    a = (player_from_tab(learner_tab, 0) if learner_id==0 else player_from_tab(opp_tab,0))(state)
                else:
                    a = (player_from_tab(opp_tab, 1) if learner_id==0 else player_from_tab(learner_tab,1))(state)
                state.apply_action(a)
        payoffs.append(state.rewards()[learner_id])
    payoffs = np.array(payoffs)
    return float(payoffs.mean()), float(payoffs.std(ddof=1))



# ---------------------------- Train code --------------------------------
def solve(
      t_max=200,
      mode: str = "selfplay",   # "selfplay" or "vs_fixed"
      pi_opp: dict[str, dict[int, float]] = None,
      eval_opp_tab=None,
      n_eval_mc=5000
      ):

  exploits = []
  vss = []
  vss_se = []
  steps = []
  gap_mean_hist = []

  infoset: Infoset = dict()

  for t in tqdm.tqdm(range(t_max)):

    # Update strategies from cumulative regret for each infonode
    update_strategy(infoset)

    if mode == "selfplay":
        for i in (0, 1):
          state = game.new_initial_state()
          cfr(infoset, state, i, t)
    elif mode == "vs_fixed":
        assert pi_opp is not None, "Must provide pi_opp for vs_fixed mode"

        for learner in (0, 1):
          state = game.new_initial_state()
          cfr_vs_fixed_opponent(infoset, state, learner, t, pi_opp)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if (t % 50 == 0) or (t == (t_max - 1)):

      # Calculate the final policy from cumulative strategy for each infonode
      update_policy(infoset)
      learner_tab = create_tabular_policy(infoset)
      e = exploitability.exploitability(game, learner_tab)
      eval_tab = eval_opp_tab if eval_opp_tab is not None else learner_tab

      root = game.new_initial_state()
      br0 = best_response.BestResponsePolicy(game, 0, eval_tab).value(root)
      br1 = best_response.BestResponsePolicy(game, 1, eval_tab).value(root)

      J0, _ = value_vs_fixed_policy(game, learner_tab, eval_tab, 0, n_eval_mc)
      J1, _ = value_vs_fixed_policy(game, learner_tab, eval_tab, 1, n_eval_mc)

      g0 = br0 - J0; g1 = br1 - J1
      gap_mean_hist.append(0.5 * (g0 + g1))

      vs, vs_se = vs_mixed(infoset, n_games=1000)

      steps.append(t)
      exploits.append(e)
      vss.append(vs)
      vss_se.append(vs_se)

  return steps, vss, vss_se, exploits, gap_mean_hist, infoset
