"""Unit tests for the environment rules. Run with:  python -m pytest"""
from env import ATTACK, BLOCK, IDLE, LEFT, RIGHT, ArenaEnv


def make(style="idle", agent_pos=4, opp_pos=5):
    """Environment with a predictable opponent and fighters placed side by side."""
    env = ArenaEnv(seed=0)
    env.reset(style)
    env.agent_pos, env.opp_pos = agent_pos, opp_pos
    return env


def test_reset_starts_at_opposite_ends_with_full_hp():
    env = ArenaEnv(seed=0)
    dist, my_hp, opp_hp, opp_last, my_rec = env.reset("idle")
    assert dist == env.size - 1
    assert my_hp == opp_hp == env.max_hp
    assert my_rec == 0


def test_attack_hits_adjacent_unblocking_opponent():
    env = make("idle")
    env.step(ATTACK)
    assert env.opp_hp == env.max_hp - 1


def test_attack_misses_when_not_adjacent():
    env = make("idle", agent_pos=2, opp_pos=6)
    env.step(ATTACK)
    assert env.opp_hp == env.max_hp


def test_block_prevents_damage():
    env = make("blocker")
    env.step(ATTACK)
    assert env.opp_hp == env.max_hp


def test_attacking_leaves_you_helpless_next_step():
    env = make("idle")
    env.step(ATTACK)                          # we attack, so we are now in recovery
    env._opponent_action = lambda: ATTACK     # opponent attacks next step
    env.step(BLOCK)                           # our BLOCK is ignored (forced idle)...
    assert env.agent_hp == env.max_hp - 1     # ...so the hit lands


def test_fighters_cannot_walk_through_each_other():
    env = make("idle")
    env.step(RIGHT)
    assert env.agent_pos == 4 and env.opp_pos == 5


def test_win_reward_and_done_when_opponent_reaches_zero_hp():
    env = make("idle")
    env.opp_hp = 1
    _, reward, done, info = env.step(ATTACK)
    assert done and info["outcome"] == "win"
    assert reward > 10   # +1 damage, +10 win bonus, -0.01 step cost


def test_timeout_ends_episode():
    env = make("idle", agent_pos=0, opp_pos=9)
    done = False
    for _ in range(env.max_steps):
        _, _, done, info = env.step(LEFT)
    assert done and info["outcome"] == "draw"
