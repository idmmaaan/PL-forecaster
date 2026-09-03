"""Club-name normalisation.

Elo and rolling form accumulate per club, so a club appearing under two names
would split its own history. These tests pin the mapping table's invariants.
"""

import pytest

from epl_predictor.data.teams import (
    CANONICAL_TEAMS,
    FOOTBALL_DATA_CO_UK_ALIASES,
    FOOTBALL_DATA_ORG_ALIASES,
    TEAM_NAME_ALIASES,
    UnknownTeamError,
    normalise_team_name,
    try_normalise_team_name,
)


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("Man United", "Manchester United"),
        ("Man City", "Manchester City"),
        ("Nott'm Forest", "Nottingham Forest"),
        ("QPR", "Queens Park Rangers"),
        ("Sheffield Weds", "Sheffield Wednesday"),
        ("Tottenham", "Tottenham Hotspur"),
        ("West Brom", "West Bromwich Albion"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("Bournemouth", "AFC Bournemouth"),
    ],
)
def test_football_data_co_uk_spellings_normalise(source_name: str, expected: str) -> None:
    assert normalise_team_name(source_name) == expected


@pytest.mark.parametrize(
    ("source_name", "expected"),
    [
        ("Arsenal FC", "Arsenal"),
        ("Manchester United FC", "Manchester United"),
        ("Nottingham Forest FC", "Nottingham Forest"),
        ("Tottenham Hotspur FC", "Tottenham Hotspur"),
        ("Brighton & Hove Albion FC", "Brighton & Hove Albion"),
        ("Wolverhampton Wanderers FC", "Wolverhampton Wanderers"),
    ],
)
def test_football_data_org_spellings_normalise(source_name: str, expected: str) -> None:
    assert normalise_team_name(source_name) == expected


def test_both_sources_agree_on_every_shared_club() -> None:
    """The whole point of the table: one club, one canonical name.

    If the API and the CSV disagreed for even one club, its history would split
    between training data and live fixtures.
    """
    csv_targets = set(FOOTBALL_DATA_CO_UK_ALIASES.values())
    api_targets = set(FOOTBALL_DATA_ORG_ALIASES.values())

    assert csv_targets == api_targets


def test_every_alias_resolves_to_a_canonical_team() -> None:
    unknown = {
        alias: target
        for alias, target in TEAM_NAME_ALIASES.items()
        if target not in CANONICAL_TEAMS
    }

    assert unknown == {}


def test_every_canonical_team_is_reachable_from_both_sources() -> None:
    csv_targets = set(FOOTBALL_DATA_CO_UK_ALIASES.values())
    api_targets = set(FOOTBALL_DATA_ORG_ALIASES.values())

    assert CANONICAL_TEAMS - csv_targets == set()
    assert CANONICAL_TEAMS - api_targets == set()


def test_normalisation_is_idempotent() -> None:
    for team in CANONICAL_TEAMS:
        assert normalise_team_name(normalise_team_name(team)) == team


def test_surrounding_and_repeated_whitespace_is_ignored() -> None:
    assert normalise_team_name("  Man   United ") == "Manchester United"


def test_an_unmapped_spelling_raises_rather_than_guessing() -> None:
    """A near-miss must fail loudly; a silent guess would corrupt club history."""
    with pytest.raises(UnknownTeamError) as excinfo:
        normalise_team_name("Manchester Utd")

    assert "Manchester Utd" in str(excinfo.value)
    assert "TEAM_NAME_ALIASES" in str(excinfo.value)


def test_normalisation_is_case_sensitive_by_design() -> None:
    """Casing differences signal a new source dialect that needs a real mapping."""
    assert try_normalise_team_name("man united") is None


def test_try_normalise_returns_none_for_unknown_names() -> None:
    assert try_normalise_team_name("Real Madrid") is None
    assert try_normalise_team_name("Arsenal") == "Arsenal"
