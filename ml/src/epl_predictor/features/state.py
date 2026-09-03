"""Chronological team state: rolling form, rest days, and league position.

This module answers one question: *what was known about a club immediately
before a given kickoff?* Everything it exposes is derived only from matches
already recorded, so a caller that records matches in kickoff order cannot
accidentally describe a club using the match it is trying to predict.

The state is deliberately write-once-per-match and read-only otherwise. The
builder enforces the README's ordering — read a snapshot, save the row, then
record the match — and the leakage tests assert it.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import date

from epl_predictor.features.elo import EloRatings

ROLLING_WINDOW = 5

# Days of rest are capped so that a three-month summer break does not dominate
# a feature whose useful signal lives in the 2-14 day range. Part of the
# feature schema contract.
REST_DAYS_CAP = 21

POINTS_FOR_WIN = 3
POINTS_FOR_DRAW = 1


def points_for(goals_for: int, goals_against: int) -> int:
    """League points earned from one result."""
    if goals_for > goals_against:
        return POINTS_FOR_WIN
    if goals_for == goals_against:
        return POINTS_FOR_DRAW
    return 0


@dataclass(frozen=True)
class TeamMatch:
    """One completed match, from a single club's point of view."""

    kickoff_date: date
    season_start_year: int
    is_home: bool
    goals_for: int
    goals_against: int
    shots_on_target_for: int | None

    @property
    def points(self) -> int:
        return points_for(self.goals_for, self.goals_against)

    @property
    def is_clean_sheet(self) -> bool:
        return self.goals_against == 0


@dataclass
class TeamForm:
    """Rolling history for one club.

    Windows extend across season boundaries: "last five matches" means the
    club's five most recent matches, which is how form is understood in
    football and avoids an undefined window every August.
    """

    window: int = ROLLING_WINDOW
    recent: deque[TeamMatch] = field(default_factory=deque)
    recent_home: deque[TeamMatch] = field(default_factory=deque)
    recent_away: deque[TeamMatch] = field(default_factory=deque)
    last_kickoff_date: date | None = None
    matches_played: int = 0

    def __post_init__(self) -> None:
        self.recent = deque(self.recent, maxlen=self.window)
        self.recent_home = deque(self.recent_home, maxlen=self.window)
        self.recent_away = deque(self.recent_away, maxlen=self.window)

    def record(self, match: TeamMatch) -> None:
        """Add a completed match to this club's history."""
        self.recent.append(match)
        (self.recent_home if match.is_home else self.recent_away).append(match)
        self.last_kickoff_date = match.kickoff_date
        self.matches_played += 1

    def rest_days(self, kickoff_date: date) -> int | None:
        """Days since this club's previous match, capped, or None if it has none."""
        if self.last_kickoff_date is None:
            return None
        return min((kickoff_date - self.last_kickoff_date).days, REST_DAYS_CAP)


@dataclass
class SeasonRecord:
    """A club's league record within one season."""

    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def record(self, goals_for: int, goals_against: int) -> None:
        self.played += 1
        self.points += points_for(goals_for, goals_against)
        self.goals_for += goals_for
        self.goals_against += goals_against


class SeasonTable:
    """League table for one season, built only from matches already played.

    Positions follow Premier League ordering: points, then goal difference,
    then goals scored. The real competition breaks a remaining tie by playoff;
    here club name is used so that positions are deterministic.
    """

    def __init__(self) -> None:
        self.records: dict[str, SeasonRecord] = {}

    def record(self, team: str, goals_for: int, goals_against: int) -> None:
        self.records.setdefault(team, SeasonRecord()).record(goals_for, goals_against)

    def positions(self) -> dict[str, int]:
        """Current one-based position of every club that has played."""
        ranked = sorted(
            self.records.items(),
            key=lambda item: (
                -item[1].points,
                -item[1].goal_difference,
                -item[1].goals_for,
                item[0],
            ),
        )
        return {team: index for index, (team, _) in enumerate(ranked, start=1)}

    def position_of(self, team: str) -> int | None:
        """Position of a club, or None before it has played in this season.

        Returning None rather than a placeholder keeps the null policy explicit:
        at the opening weekend nobody has a league position, and inventing one
        would put a fabricated value into every model that consumes it.
        """
        if team not in self.records:
            return None
        return self.positions().get(team)


@dataclass(frozen=True)
class TeamSnapshot:
    """Everything known about one club immediately before a kickoff.

    A `None` means the club genuinely has no value yet — fewer than five
    matches of history, or no match at all this season — rather than a zero.
    Collapsing those cases into 0 would tell a model that a debutant had lost
    its last five matches.
    """

    team: str
    elo: float
    matches_played: int
    points_last_5: int | None
    goals_for_last_5: int | None
    goals_against_last_5: int | None
    goal_difference_last_5: int | None
    venue_points_last_5: int | None
    shots_on_target_last_5: int | None
    clean_sheet_rate_last_5: float | None
    rest_days: int | None
    league_position: int | None
    season_matches_played: int


def _full_window(matches: "deque[TeamMatch] | list[TeamMatch]", window: int) -> bool:
    """True when the rolling window is complete.

    A partial window is reported as null rather than summed: three points from
    two matches and three from five describe very different clubs, and summing
    them into the same number would be silently misleading.
    """
    return len(matches) >= window


class LeagueState:
    """Chronological state for every club in the league.

    Read a snapshot for a fixture, then record that fixture. Reading after
    recording would describe a club using the very match being predicted,
    which is the failure mode the README calls the project's largest risk.
    """

    def __init__(self, window: int = ROLLING_WINDOW, elo: "EloRatings | None" = None):
        self.window = window
        self.forms: dict[str, TeamForm] = {}
        self.season_tables: dict[int, SeasonTable] = {}
        self.elo = elo if elo is not None else EloRatings()
        self._current_season: int | None = None

    def form(self, team: str) -> TeamForm:
        return self.forms.setdefault(team, TeamForm(window=self.window))

    def season_table(self, season_start_year: int) -> SeasonTable:
        return self.season_tables.setdefault(season_start_year, SeasonTable())

    def begin_season(self, season_start_year: int) -> None:
        """Regress Elo when the state first reaches a new season.

        Called by the builder as it walks matches in order; repeated calls for
        the same season are ignored, and going backwards is refused because it
        would mean the input was not chronological.
        """
        if self._current_season == season_start_year:
            return
        if self._current_season is not None:
            if season_start_year < self._current_season:
                raise ValueError(
                    f"Season {season_start_year} follows {self._current_season}: "
                    "matches must be processed in chronological order."
                )
            self.elo.apply_season_regression()
        self._current_season = season_start_year

    def snapshot(
        self, team: str, kickoff_date: date, season_start_year: int, is_home: bool
    ) -> TeamSnapshot:
        """Describe a club using only matches recorded so far."""
        form = self.form(team)
        table = self.season_table(season_start_year)
        recent = list(form.recent)
        venue = list(form.recent_home if is_home else form.recent_away)

        complete = _full_window(recent, self.window)
        goals_for = sum(match.goals_for for match in recent) if complete else None
        goals_against = sum(match.goals_against for match in recent) if complete else None

        shots = [m.shots_on_target_for for m in recent if m.shots_on_target_for is not None]
        record = table.records.get(team)

        return TeamSnapshot(
            team=team,
            elo=self.elo.rating(team),
            matches_played=form.matches_played,
            points_last_5=sum(match.points for match in recent) if complete else None,
            goals_for_last_5=goals_for,
            goals_against_last_5=goals_against,
            goal_difference_last_5=(
                goals_for - goals_against
                if goals_for is not None and goals_against is not None
                else None
            ),
            venue_points_last_5=(
                sum(match.points for match in venue) if _full_window(venue, self.window) else None
            ),
            shots_on_target_last_5=(sum(shots) if complete and len(shots) == len(recent) else None),
            clean_sheet_rate_last_5=(
                sum(match.is_clean_sheet for match in recent) / len(recent) if complete else None
            ),
            rest_days=form.rest_days(kickoff_date),
            league_position=table.position_of(team),
            season_matches_played=record.played if record else 0,
        )

    def record_match(
        self,
        kickoff_date: date,
        season_start_year: int,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        home_shots_on_target: int | None = None,
        away_shots_on_target: int | None = None,
    ) -> None:
        """Fold one completed match into every part of the state."""
        self.form(home_team).record(
            TeamMatch(
                kickoff_date=kickoff_date,
                season_start_year=season_start_year,
                is_home=True,
                goals_for=home_goals,
                goals_against=away_goals,
                shots_on_target_for=home_shots_on_target,
            )
        )
        self.form(away_team).record(
            TeamMatch(
                kickoff_date=kickoff_date,
                season_start_year=season_start_year,
                is_home=False,
                goals_for=away_goals,
                goals_against=home_goals,
                shots_on_target_for=away_shots_on_target,
            )
        )

        table = self.season_table(season_start_year)
        table.record(home_team, home_goals, away_goals)
        table.record(away_team, away_goals, home_goals)

        self.elo.update(home_team, away_team, home_goals, away_goals)
