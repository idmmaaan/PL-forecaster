"""Chronological team state: rolling form, rest days, and league position.

The null policy is the subject of most of these tests. A club with two matches
of history has no five-match form, and saying so with a null is materially
different from claiming it took zero points from five games.
"""

from datetime import date, timedelta

import pytest

from epl_predictor.features.state import (
    REST_DAYS_CAP,
    ROLLING_WINDOW,
    LeagueState,
    SeasonTable,
    TeamForm,
    TeamMatch,
    points_for,
)

SEASON = 2024
KICKOFF = date(2024, 8, 17)


def team_match(
    goals_for: int,
    goals_against: int,
    day_offset: int = 0,
    is_home: bool = True,
    shots_on_target: int | None = 5,
) -> TeamMatch:
    return TeamMatch(
        kickoff_date=KICKOFF + timedelta(days=day_offset),
        season_start_year=SEASON,
        is_home=is_home,
        goals_for=goals_for,
        goals_against=goals_against,
        shots_on_target_for=shots_on_target,
    )


@pytest.mark.parametrize(
    ("goals_for", "goals_against", "points"), [(2, 0, 3), (1, 1, 1), (0, 3, 0)]
)
def test_points_follow_the_league_system(goals_for: int, goals_against: int, points: int) -> None:
    assert points_for(goals_for, goals_against) == points


def test_a_match_reports_its_own_points_and_clean_sheet() -> None:
    win = team_match(3, 0)

    assert win.points == 3
    assert win.is_clean_sheet

    loss = team_match(0, 1)
    assert loss.points == 0
    assert not loss.is_clean_sheet


# --- Rolling form -----------------------------------------------------------


def test_form_keeps_only_the_most_recent_window() -> None:
    form = TeamForm()
    for offset in range(8):
        form.record(team_match(1, 0, day_offset=offset * 3))

    assert len(form.recent) == ROLLING_WINDOW
    assert form.matches_played == 8


def test_home_and_away_histories_are_tracked_separately() -> None:
    form = TeamForm()
    form.record(team_match(2, 0, is_home=True))
    form.record(team_match(0, 2, day_offset=3, is_home=False))

    assert len(form.recent_home) == 1
    assert len(form.recent_away) == 1
    assert form.recent_home[0].points == 3
    assert form.recent_away[0].points == 0


def test_rest_days_measure_the_gap_since_the_previous_match() -> None:
    form = TeamForm()
    form.record(team_match(1, 0, day_offset=0))

    assert form.rest_days(KICKOFF + timedelta(days=4)) == 4


def test_rest_days_are_capped_so_the_summer_break_does_not_dominate() -> None:
    form = TeamForm()
    form.record(team_match(1, 0, day_offset=0))

    assert form.rest_days(KICKOFF + timedelta(days=90)) == REST_DAYS_CAP


def test_rest_days_are_unknown_before_a_clubs_first_match() -> None:
    assert TeamForm().rest_days(KICKOFF) is None


# --- League table -----------------------------------------------------------


def test_the_table_orders_by_points_then_goal_difference_then_goals_scored() -> None:
    table = SeasonTable()
    table.record("Leader", 3, 0)  # 3 points, +3
    table.record("Second", 2, 0)  # 3 points, +2
    table.record("Third", 2, 1)  # 3 points, +1
    table.record("Fourth", 0, 0)  # 1 point

    assert table.positions() == {"Leader": 1, "Second": 2, "Third": 3, "Fourth": 4}


def test_goals_scored_breaks_an_equal_goal_difference() -> None:
    table = SeasonTable()
    table.record("Prolific", 3, 1)  # +2, 3 scored
    table.record("Frugal", 2, 0)  # +2, 2 scored

    assert table.positions()["Prolific"] == 1


def test_a_club_has_no_position_until_it_has_played() -> None:
    """Inventing an opening-day position would fabricate a feature value."""
    table = SeasonTable()
    table.record("Arsenal", 1, 0)

    assert table.position_of("Arsenal") == 1
    assert table.position_of("Chelsea") is None


def test_the_table_accumulates_across_matches() -> None:
    table = SeasonTable()
    table.record("Arsenal", 3, 0)
    table.record("Arsenal", 1, 1)

    record = table.records["Arsenal"]

    assert (record.played, record.points, record.goal_difference) == (2, 4, 3)


# --- Snapshots --------------------------------------------------------------


def play(state: LeagueState, home: str, away: str, hg: int, ag: int, offset: int) -> None:
    state.record_match(
        kickoff_date=KICKOFF + timedelta(days=offset),
        season_start_year=SEASON,
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        home_shots_on_target=5,
        away_shots_on_target=3,
    )


def test_a_debut_snapshot_is_null_rather_than_zero() -> None:
    """The distinction a model needs: "no history" is not "no points"."""
    snapshot = LeagueState().snapshot("Arsenal", KICKOFF, SEASON, is_home=True)

    assert snapshot.matches_played == 0
    assert snapshot.points_last_5 is None
    assert snapshot.goals_for_last_5 is None
    assert snapshot.goal_difference_last_5 is None
    assert snapshot.clean_sheet_rate_last_5 is None
    assert snapshot.rest_days is None
    assert snapshot.league_position is None
    assert snapshot.elo == 1500.0


def test_a_partial_window_stays_null() -> None:
    """Four points from two games is not comparable with four from five."""
    state = LeagueState()
    for index in range(ROLLING_WINDOW - 1):
        play(state, "Arsenal", f"Rival {index}", 1, 0, offset=index * 3)

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=30), SEASON, is_home=True)

    assert snapshot.matches_played == ROLLING_WINDOW - 1
    assert snapshot.points_last_5 is None
    assert snapshot.league_position == 1


def test_a_complete_window_reports_rolling_totals() -> None:
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        play(state, "Arsenal", f"Rival {index}", 2, 1, offset=index * 3)

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=30), SEASON, is_home=True)

    assert snapshot.points_last_5 == 15
    assert snapshot.goals_for_last_5 == 10
    assert snapshot.goals_against_last_5 == 5
    assert snapshot.goal_difference_last_5 == 5
    assert snapshot.clean_sheet_rate_last_5 == 0.0
    assert snapshot.shots_on_target_last_5 == 25


def test_clean_sheet_rate_is_a_proportion() -> None:
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        conceded = 0 if index < 3 else 2
        play(state, "Arsenal", f"Rival {index}", 1, conceded, offset=index * 3)

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=30), SEASON, is_home=True)

    assert snapshot.clean_sheet_rate_last_5 == pytest.approx(0.6)


def test_venue_points_count_only_matches_at_that_venue() -> None:
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        play(state, "Arsenal", f"Rival {index}", 1, 0, offset=index * 6)
    for index in range(ROLLING_WINDOW):
        play(state, f"Other {index}", "Arsenal", 3, 0, offset=index * 6 + 3)

    at_home = state.snapshot("Arsenal", KICKOFF + timedelta(days=60), SEASON, is_home=True)
    away = state.snapshot("Arsenal", KICKOFF + timedelta(days=60), SEASON, is_home=False)

    assert at_home.venue_points_last_5 == 15
    assert away.venue_points_last_5 == 0


def test_venue_points_stay_null_until_five_matches_at_that_venue() -> None:
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        play(state, "Arsenal", f"Rival {index}", 1, 0, offset=index * 3)

    away = state.snapshot("Arsenal", KICKOFF + timedelta(days=30), SEASON, is_home=False)

    assert away.venue_points_last_5 is None


def test_shots_are_null_when_any_match_in_the_window_lacks_them() -> None:
    """Older seasons have no shot data; a partial sum would understate the club."""
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        state.record_match(
            kickoff_date=KICKOFF + timedelta(days=index * 3),
            season_start_year=SEASON,
            home_team="Arsenal",
            away_team=f"Rival {index}",
            home_goals=1,
            away_goals=0,
            home_shots_on_target=None if index == 0 else 5,
        )

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=30), SEASON, is_home=True)

    assert snapshot.points_last_5 == 15
    assert snapshot.shots_on_target_last_5 is None


def test_the_rolling_window_slides_past_older_matches() -> None:
    state = LeagueState()
    play(state, "Arsenal", "Early", 5, 0, offset=0)
    for index in range(ROLLING_WINDOW):
        play(state, "Arsenal", f"Rival {index}", 0, 1, offset=(index + 1) * 3)

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=40), SEASON, is_home=True)

    assert snapshot.points_last_5 == 0
    assert snapshot.goals_for_last_5 == 0
    assert snapshot.matches_played == ROLLING_WINDOW + 1


def test_form_carries_across_a_season_boundary() -> None:
    """ "Last five matches" means the club's last five, as football uses the term."""
    state = LeagueState()
    for index in range(ROLLING_WINDOW):
        play(state, "Arsenal", f"Rival {index}", 2, 0, offset=index * 3)

    state.begin_season(SEASON + 1)
    snapshot = state.snapshot("Arsenal", date(2025, 8, 16), SEASON + 1, is_home=True)

    assert snapshot.points_last_5 == 15
    assert snapshot.league_position is None, "the new season's table starts empty"
    assert snapshot.season_matches_played == 0


def test_beginning_a_new_season_regresses_elo() -> None:
    state = LeagueState()
    play(state, "Arsenal", "Chelsea", 4, 0, offset=0)
    peak = state.elo.rating("Arsenal")

    state.begin_season(SEASON)
    state.begin_season(SEASON + 1)

    assert state.elo.rating("Arsenal") < peak


def test_re_entering_the_same_season_does_not_regress_twice() -> None:
    state = LeagueState()
    play(state, "Arsenal", "Chelsea", 4, 0, offset=0)
    state.begin_season(SEASON + 1)
    once = state.elo.rating("Arsenal")

    state.begin_season(SEASON + 1)

    assert state.elo.rating("Arsenal") == once


def test_going_back_to_an_earlier_season_is_refused() -> None:
    """Out-of-order seasons would mean the caller is not walking chronologically."""
    state = LeagueState()
    state.begin_season(SEASON)

    with pytest.raises(ValueError, match="chronological"):
        state.begin_season(SEASON - 1)


def test_recording_a_match_updates_both_clubs() -> None:
    state = LeagueState()
    play(state, "Arsenal", "Chelsea", 2, 0, offset=0)

    home = state.snapshot("Arsenal", KICKOFF + timedelta(days=7), SEASON, is_home=True)
    away = state.snapshot("Chelsea", KICKOFF + timedelta(days=7), SEASON, is_home=False)

    assert (home.matches_played, away.matches_played) == (1, 1)
    assert home.league_position == 1
    assert away.league_position == 2
    assert home.elo > away.elo
    assert home.rest_days == 7


def test_season_matches_played_drives_the_matchday_derivation() -> None:
    state = LeagueState()
    for index in range(3):
        play(state, "Arsenal", f"Rival {index}", 1, 0, offset=index * 3)

    snapshot = state.snapshot("Arsenal", KICKOFF + timedelta(days=20), SEASON, is_home=True)

    assert snapshot.season_matches_played == 3
