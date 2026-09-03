"""Elo rating behaviour.

Elo is the strongest single feature in the v1 set, so its constants and update
rule are pinned here. Changing any of them changes the meaning of `home_elo`
and requires a new feature schema version.
"""

import pytest

from epl_predictor.features.elo import (
    HOME_ADVANTAGE,
    INITIAL_RATING,
    EloRatings,
    actual_home_score,
    expected_home_score,
    goal_difference_multiplier,
)


def test_equal_ratings_still_favour_the_home_side() -> None:
    """Home advantage is the whole reason a neutral fixture is not 50/50."""
    expected = expected_home_score(INITIAL_RATING, INITIAL_RATING)

    assert expected > 0.5
    assert expected == pytest.approx(0.585, abs=0.005)


def test_removing_home_advantage_gives_an_even_fixture() -> None:
    assert expected_home_score(1500, 1500, home_advantage=0) == pytest.approx(0.5)


def test_a_stronger_side_is_expected_to_score_higher() -> None:
    strong_at_home = expected_home_score(1800, 1400)
    strong_away = expected_home_score(1400, 1800)

    assert strong_at_home > 0.9
    assert strong_away < 0.15
    assert strong_at_home > strong_away


def test_swapping_the_venue_mirrors_the_expectation() -> None:
    """Both sides' expectations sum to one once home advantage swaps with them."""
    home = expected_home_score(1700, 1450, home_advantage=HOME_ADVANTAGE)
    away = expected_home_score(1450, 1700, home_advantage=-HOME_ADVANTAGE)

    assert home + away == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("gap", "expected"), [(0, 0.585), (100, 0.715), (-100, 0.443), (400, 0.934)]
)
def test_expectation_rises_monotonically_with_the_rating_gap(gap: int, expected: float) -> None:
    assert expected_home_score(1500 + gap, 1500) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize(
    ("home", "away", "score"), [(2, 0, 1.0), (0, 2, 0.0), (1, 1, 0.5), (0, 0, 0.5)]
)
def test_result_maps_to_the_elo_score_scale(home: int, away: int, score: float) -> None:
    assert actual_home_score(home, away) == score


@pytest.mark.parametrize(
    ("margin", "multiplier"),
    [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (4, 1.875), (6, 2.125)],
)
def test_wider_margins_move_ratings_further(margin: int, multiplier: float) -> None:
    assert goal_difference_multiplier(margin, 0) == pytest.approx(multiplier)


def test_the_multiplier_ignores_which_side_won() -> None:
    assert goal_difference_multiplier(4, 0) == goal_difference_multiplier(0, 4)


def test_an_unseen_club_starts_at_the_mean() -> None:
    assert EloRatings().rating("Brentford") == INITIAL_RATING


def test_a_home_win_moves_both_ratings_in_opposite_directions() -> None:
    elo = EloRatings()
    elo.update("Arsenal", "Chelsea", 2, 0)

    assert elo.rating("Arsenal") > INITIAL_RATING
    assert elo.rating("Chelsea") < INITIAL_RATING


def test_elo_is_zero_sum() -> None:
    """The winner's gain is exactly the loser's loss, so the mean is preserved."""
    elo = EloRatings()
    elo.update("Arsenal", "Chelsea", 3, 1)

    total = elo.rating("Arsenal") + elo.rating("Chelsea")

    assert total == pytest.approx(2 * INITIAL_RATING)


def test_an_expected_home_win_moves_ratings_less_than_an_upset() -> None:
    favourite = EloRatings(ratings={"Strong": 1800.0, "Weak": 1200.0})
    favourite.update("Strong", "Weak", 1, 0)
    expected_gain = favourite.rating("Strong") - 1800.0

    upset = EloRatings(ratings={"Strong": 1800.0, "Weak": 1200.0})
    upset.update("Weak", "Strong", 1, 0)
    upset_gain = upset.rating("Weak") - 1200.0

    assert 0 < expected_gain < upset_gain


def test_a_draw_costs_the_favourite_rating_points() -> None:
    elo = EloRatings(ratings={"Strong": 1800.0, "Weak": 1200.0})
    elo.update("Strong", "Weak", 1, 1)

    assert elo.rating("Strong") < 1800.0
    assert elo.rating("Weak") > 1200.0


def test_a_heavier_win_earns_more_than_a_narrow_one() -> None:
    narrow = EloRatings()
    narrow.update("Arsenal", "Chelsea", 1, 0)

    thrashing = EloRatings()
    thrashing.update("Arsenal", "Chelsea", 5, 0)

    assert thrashing.rating("Arsenal") > narrow.rating("Arsenal")


def test_season_regression_pulls_ratings_toward_the_mean() -> None:
    """A club's May rating overstates what is known the following August."""
    elo = EloRatings(ratings={"Strong": 1700.0, "Weak": 1300.0})

    elo.apply_season_regression()

    assert elo.rating("Strong") == pytest.approx(1650.0)
    assert elo.rating("Weak") == pytest.approx(1350.0)


def test_season_regression_preserves_the_league_mean() -> None:
    elo = EloRatings(ratings={"A": 1700.0, "B": 1300.0, "C": 1500.0})
    before = sum(elo.as_dict().values())

    elo.apply_season_regression()

    assert sum(elo.as_dict().values()) == pytest.approx(before)


def test_season_regression_can_be_switched_off() -> None:
    elo = EloRatings(season_regression=0.0, ratings={"Strong": 1700.0})

    elo.apply_season_regression()

    assert elo.rating("Strong") == 1700.0


def test_regression_never_crosses_the_mean() -> None:
    elo = EloRatings(season_regression=1.0, ratings={"Strong": 1700.0})

    elo.apply_season_regression()

    assert elo.rating("Strong") == pytest.approx(INITIAL_RATING)


def test_a_copy_is_independent_of_its_source() -> None:
    """Inference must be able to evaluate a fixture without mutating state."""
    elo = EloRatings(ratings={"Arsenal": 1600.0})
    snapshot = elo.copy()

    elo.update("Arsenal", "Chelsea", 3, 0)

    assert snapshot.rating("Arsenal") == 1600.0
    assert elo.rating("Arsenal") > 1600.0


def test_as_dict_returns_a_detached_table() -> None:
    elo = EloRatings(ratings={"Arsenal": 1600.0})
    exported = elo.as_dict()
    exported["Arsenal"] = 0.0

    assert elo.rating("Arsenal") == 1600.0


def test_expected_matches_the_ratings_it_holds() -> None:
    elo = EloRatings(ratings={"Arsenal": 1700.0, "Chelsea": 1500.0})

    assert elo.expected("Arsenal", "Chelsea") == expected_home_score(1700.0, 1500.0)


def test_ratings_converge_toward_true_strength_over_a_season() -> None:
    """A club that always wins must end up rated above one that always loses."""
    elo = EloRatings()
    for _ in range(19):
        elo.update("Winner", "Loser", 2, 0)
        elo.update("Loser", "Winner", 0, 2)

    assert elo.rating("Winner") > 1700
    assert elo.rating("Loser") < 1300
