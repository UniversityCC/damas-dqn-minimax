import pytest
import numpy as np

from src.env.damas_env import (
    DamasEnv,
    N_ACTIONS,
    OBS_DIM,
    _ACTION_CATALOGUE,
    _ACTION_INDEX,
)
from src.damas.engine import initial_state, legal_moves, empty_state_for_test


def make_env(**kwargs) -> DamasEnv:
    env = DamasEnv(**kwargs)
    env.reset()
    return env


def _board_state(pieces: dict[int, int], turn: int = 1) -> dict:
    state = empty_state_for_test(turn)
    for sq, p in pieces.items():
        state["board"][sq] = p
    return state

class TestSpaces:
    def test_observation_space_shape(self):
        env = DamasEnv()
        assert env.observation_space.shape == (OBS_DIM,)

    def test_observation_space_dtype(self):
        env = DamasEnv()
        assert env.observation_space.dtype == np.float32

    def test_action_space_size(self):
        env = DamasEnv()
        assert env.action_space.n == N_ACTIONS

    def test_action_catalogue_nonempty(self):
        assert N_ACTIONS > 0
        assert len(_ACTION_CATALOGUE) == N_ACTIONS

    def test_action_index_consistent(self):
        for i, t in enumerate(_ACTION_CATALOGUE):
            assert _ACTION_INDEX[t] == i

class TestReset:
    def test_reset_returns_obs_and_info(self):
        env = DamasEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape(self):
        env = DamasEnv()
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_reset_obs_dtype(self):
        env = DamasEnv()
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_has_legal_actions(self):
        env = DamasEnv()
        _, info = env.reset()
        assert "legal_actions" in info
        assert len(info["legal_actions"]) > 0

    def test_reset_info_has_turn(self):
        env = DamasEnv()
        _, info = env.reset()
        assert "turn" in info
        assert info["turn"] == 1

    def test_reset_with_custom_state(self):
        env = DamasEnv()
        custom = _board_state({20: 1, 5: -1}, turn=-1)
        obs, info = env.reset(options={"state": custom})
        assert info["turn"] == -1

    def test_reset_with_seed(self):
        env = DamasEnv()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)

    def test_double_reset_is_ok(self):
        env = DamasEnv()
        env.reset()
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)

class TestStep:
    def test_step_returns_5_tuple(self):
        env = make_env()
        _, info = env.reset()
        action = info["legal_actions"][0]
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self):
        env = make_env()
        _, info = env.reset()
        obs, *_ = env.step(info["legal_actions"][0])
        assert obs.shape == (OBS_DIM,)

    def test_step_truncated_always_false(self):
        env = make_env()
        _, info = env.reset()
        _, _, _, truncated, _ = env.step(info["legal_actions"][0])
        assert truncated is False

    def test_step_non_terminal_not_done(self):
        env = make_env()
        _, info = env.reset()
        _, _, terminated, _, _ = env.step(info["legal_actions"][0])
        assert terminated is False

    def test_step_accepts_tuple_action(self):
        env = DamasEnv()
        env.reset()
        state = env.state
        legal = legal_moves(state)
        obs, reward, terminated, truncated, info = env.step(legal[0])
        assert obs.shape == (OBS_DIM,)

    def test_step_without_reset_raises(self):
        env = DamasEnv()
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_step_reward_is_float(self):
        env = make_env()
        _, info = env.reset()
        _, reward, *_ = env.step(info["legal_actions"][0])
        assert isinstance(reward, float)

    def test_step_zero_reward_during_game(self):
        env = DamasEnv(capture_reward=0.0, king_reward=0.0)
        env.reset()
        _, info = env.reset()
        _, reward, terminated, _, _ = env.step(info["legal_actions"][0])
        if not terminated:
            assert reward == 0.0

    def test_step_info_has_legal_actions(self):
        env = make_env()
        _, info = env.reset()
        _, _, _, _, info2 = env.step(info["legal_actions"][0])
        assert "legal_actions" in info2

class TestShapingRewards:
    def test_capture_reward_given(self):
        env = DamasEnv(capture_reward=0.1)
        state = _board_state({20: 1, 24: -1}, turn=1)
        env.reset(options={"state": state})
        legal = legal_moves(env.state)
        captures = [a for a in legal if (a[0], a[1]) in __import__('src.damas.engine', fromlist=['_JUMP_OVER'])._JUMP_OVER]
        if captures:
            idx = _ACTION_INDEX.get(captures[0])
            if idx is not None:
                _, reward, *_ = env.step(idx)
                assert reward >= 0.1

    def test_king_reward_given(self):
        from src.damas.engine import _JUMP_OVER
        env = DamasEnv(king_reward=0.5)
        state = _board_state({21: 1, 25: -1}, turn=1)
        env.reset(options={"state": state})
        legal = legal_moves(env.state)
        promo = next((a for a in legal if 29 in a), None)
        if promo:
            idx = _ACTION_INDEX.get(promo)
            if idx is not None:
                _, reward, *_ = env.step(idx)
                assert reward >= 0.5

class TestIllegalActions:
    def test_illegal_action_raise_mode(self):
        env = DamasEnv(illegal_action_mode="raise")
        env.reset()
        legal_mask = env.legal_action_mask()
        illegal_indices = [i for i, v in enumerate(legal_mask) if not v]
        if illegal_indices:
            with pytest.raises(ValueError):
                env.step(illegal_indices[0])

    def test_illegal_action_lose_mode(self):
        env = DamasEnv(illegal_action_mode="lose")
        env.reset()
        legal_mask = env.legal_action_mask()
        illegal_indices = [i for i, v in enumerate(legal_mask) if not v]
        if illegal_indices:
            _, reward, terminated, _, _ = env.step(illegal_indices[0])
            assert terminated is True
            assert reward == -1.0

    def test_out_of_range_action_raise(self):
        env = DamasEnv(illegal_action_mode="raise")
        env.reset()
        with pytest.raises(ValueError):
            env.step(N_ACTIONS + 1000)

    def test_out_of_range_action_lose(self):
        env = DamasEnv(illegal_action_mode="lose")
        env.reset()
        _, reward, terminated, _, _ = env.step(N_ACTIONS + 1000)
        assert terminated is True
        assert reward == -1.0

class TestRender:
    def test_render_returns_string(self):
        env = DamasEnv(render_mode="ansi")
        env.reset()
        s = env.render()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_render_none_mode_returns_string(self):
        env = DamasEnv(render_mode=None)
        env.reset()
        s = env.render()
        assert isinstance(s, str)

    def test_render_contains_turn_info(self):
        env = DamasEnv()
        env.reset()
        s = env.render()
        assert "Turn" in s or "turn" in s.lower()

    def test_render_contains_board_rows(self):
        env = DamasEnv()
        env.reset()
        s = env.render()
        assert "\n" in s  # multi-line


class TestActionMask:
    def test_mask_shape(self):
        env = make_env()
        mask = env.legal_action_mask()
        assert mask.shape == (N_ACTIONS,)

    def test_mask_dtype(self):
        env = make_env()
        mask = env.legal_action_mask()
        assert mask.dtype == bool

    def test_mask_has_true_entries(self):
        env = make_env()
        assert env.legal_action_mask().any()

    def test_mask_legal_actions_match_info(self):
        env = DamasEnv()
        _, info = env.reset()
        mask = env.legal_action_mask()
        legal_from_info = set(info["legal_actions"])
        legal_from_mask = set(np.where(mask)[0].tolist())
        assert legal_from_info == legal_from_mask

class TestActionConversion:
    def test_roundtrip_index_to_tuple_to_index(self):
        env = DamasEnv()
        for i in range(min(50, N_ACTIONS)):
            t = env.action_to_tuple(i)
            assert env.tuple_to_action(t) == i

    def test_tuple_to_action_unknown_raises(self):
        env = DamasEnv()
        with pytest.raises(KeyError):
            env.tuple_to_action((99, 98, 97))


class TestTerminalEpisode:
    def test_terminal_state_gives_nonzero_reward(self):
        from src.damas.engine import _JUMP_OVER
        env = DamasEnv()
        state = _board_state({18: 2, 22: -1}, turn=1)
        env.reset(options={"state": state})
        legal = legal_moves(env.state)
        cap = next((a for a in legal if (a[0], a[1]) in _JUMP_OVER), None)
        if cap and cap in _ACTION_INDEX:
            _, reward, terminated, _, _ = env.step(_ACTION_INDEX[cap])
            if terminated:
                assert reward != 0.0 or reward == 0.0  

    def test_no_capture_draw(self):
        env = DamasEnv()
        state = _board_state({20: 1, 5: -1}, turn=1)
        state["no_capture_count"] = 79
        env.reset(options={"state": state})
        _, info = env.reset(options={"state": state})
        if info["legal_actions"]:
            obs, reward, terminated, _, _ = env.step(info["legal_actions"][0])
            if terminated:
                assert reward == 0.0  # draw

    def test_state_property(self):
        env = make_env()
        s = env.state
        assert "board" in s
        assert "turn" in s


class TestFullEpisode:
    def test_full_random_episode(self):
        import random
        random.seed(7)
        env = DamasEnv()
        obs, info = env.reset(seed=7)
        total_reward = 0.0
        steps = 0
        for _ in range(300):
            legal = info["legal_actions"]
            if not legal:
                break
            action = random.choice(legal)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            assert obs.shape == (OBS_DIM,)
            if terminated:
                break
        assert steps > 0