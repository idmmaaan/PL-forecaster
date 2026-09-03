"""Canonical club names and the source spellings that map onto them.

Two sources name the same clubs differently: Football-Data.co.uk writes
"Man United" and "Nott'm Forest", while football-data.org writes "Manchester
United FC" and "Nottingham Forest FC". Elo ratings and rolling form are
accumulated per club, so a single club appearing under two names would split
its history in half and quietly corrupt every feature derived from it.

Normalisation is therefore a lookup, never a fuzzy match: a spelling that is
not in this table is rejected with a clear error rather than being guessed at.
"""

CANONICAL_TEAMS: frozenset[str] = frozenset(
    {
        "AFC Bournemouth",
        "Arsenal",
        "Aston Villa",
        "Birmingham City",
        "Blackburn Rovers",
        "Blackpool",
        "Bolton Wanderers",
        "Brentford",
        "Brighton & Hove Albion",
        "Burnley",
        "Cardiff City",
        "Chelsea",
        "Crystal Palace",
        "Everton",
        "Fulham",
        "Huddersfield Town",
        "Hull City",
        "Ipswich Town",
        "Leeds United",
        "Leicester City",
        "Liverpool",
        "Luton Town",
        "Manchester City",
        "Manchester United",
        "Middlesbrough",
        "Newcastle United",
        "Norwich City",
        "Nottingham Forest",
        "Portsmouth",
        "Queens Park Rangers",
        "Reading",
        "Sheffield United",
        "Sheffield Wednesday",
        "Southampton",
        "Stoke City",
        "Sunderland",
        "Swansea City",
        "Tottenham Hotspur",
        "Watford",
        "West Bromwich Albion",
        "West Ham United",
        "Wigan Athletic",
        "Wolverhampton Wanderers",
    }
)

# Football-Data.co.uk CSV spellings (the historical training source).
FOOTBALL_DATA_CO_UK_ALIASES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Birmingham": "Birmingham City",
    "Blackburn": "Blackburn Rovers",
    "Blackpool": "Blackpool",
    "Bolton": "Bolton Wanderers",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton & Hove Albion",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff City",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Liverpool": "Liverpool",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Portsmouth": "Portsmouth",
    "QPR": "Queens Park Rangers",
    "Reading": "Reading",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Southampton": "Southampton",
    "Stoke": "Stoke City",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea City",
    "Tottenham": "Tottenham Hotspur",
    "Watford": "Watford",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wigan": "Wigan Athletic",
    "Wolves": "Wolverhampton Wanderers",
}

# football-data.org v4 spellings (the live fixture source), so a fixture coming
# from the API resolves to the same club as its historical CSV rows.
FOOTBALL_DATA_ORG_ALIASES: dict[str, str] = {
    "AFC Bournemouth": "AFC Bournemouth",
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Birmingham City FC": "Birmingham City",
    "Blackburn Rovers FC": "Blackburn Rovers",
    "Blackpool FC": "Blackpool",
    "Bolton Wanderers FC": "Bolton Wanderers",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff City",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Huddersfield Town AFC": "Huddersfield Town",
    "Hull City AFC": "Hull City",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton Town",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Middlesbrough FC": "Middlesbrough",
    "Newcastle United FC": "Newcastle United",
    "Norwich City FC": "Norwich City",
    "Nottingham Forest FC": "Nottingham Forest",
    "Portsmouth FC": "Portsmouth",
    "Queens Park Rangers FC": "Queens Park Rangers",
    "Reading FC": "Reading",
    "Sheffield United FC": "Sheffield United",
    "Sheffield Wednesday FC": "Sheffield Wednesday",
    "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke City",
    "Sunderland AFC": "Sunderland",
    "Swansea City AFC": "Swansea City",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "West Ham United FC": "West Ham United",
    "Wigan Athletic FC": "Wigan Athletic",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
}

TEAM_NAME_ALIASES: dict[str, str] = {
    **FOOTBALL_DATA_CO_UK_ALIASES,
    **FOOTBALL_DATA_ORG_ALIASES,
    # Canonical names map to themselves, so normalising twice is a no-op.
    **{name: name for name in CANONICAL_TEAMS},
}


class UnknownTeamError(KeyError):
    """A club spelling is absent from the alias table."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(
            f"Unknown team name {name!r}. Add it to TEAM_NAME_ALIASES in "
            "epl_predictor.data.teams rather than guessing a match."
        )


def normalise_team_name(name: str) -> str:
    """Map a source club spelling onto its canonical name.

    Raises:
        UnknownTeamError: The spelling is not in the alias table.
    """
    cleaned = " ".join(str(name).split())
    try:
        return TEAM_NAME_ALIASES[cleaned]
    except KeyError:
        raise UnknownTeamError(cleaned) from None


def try_normalise_team_name(name: str) -> str | None:
    """Like `normalise_team_name`, but returns None instead of raising."""
    try:
        return normalise_team_name(name)
    except UnknownTeamError:
        return None
