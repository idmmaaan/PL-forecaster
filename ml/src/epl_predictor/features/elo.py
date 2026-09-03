"""Elo ratings for football clubs.

Elo compresses a club's whole result history into one number, which makes it
the most informative pre-kickoff feature available from score data alone. The
implementation follows the World Football Elo conventions: a rating-point home
advantage, and a K-factor scaled by goal difference so a 4-0 moves the ratings
further than a 1-0.

Ratings are regressed toward the mean between seasons, because squads change
over a summer and a club's rating from May overstates what is known in August.

Every constant here is part of the feature schema contract: changing any of
them changes the meaning of `home_elo` and `away_elo`, and so requires a new
`feature_schema_version`.
"""

from dataclasses import dataclass, field

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 60.0
SEASON_REGRESSION = 0.25
RATING_SCALE = 400.0


def expected_home_score(
    home_rating: float, away_rating: float, home_advantage: float = HOME_ADVANTAGE
) -> float:
    """Expected score of the home side in [0, 1].

    A draw counts as half, so this is an expected points share rather than a
    win probability.
    """
    difference = (home_rating + home_advantage) - away_rating
    return 1.0 / (1.0 + 10.0 ** (-difference / RATING_SCALE))


def actual_home_score(home_goals: int, away_goals: int) -> float:
    """Map a result onto Elo's 1 / 0.5 / 0 scale, from the home side's view."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def goal_difference_multiplier(home_goals: int, away_goals: int) -> float:
    """Scale K by margin of victory, per the World Football Elo formula.

    A one-goal game moves ratings by the base amount; wider margins move them
    further, with diminishing returns above a three-goal win.
    """
    margin = abs(home_goals - away_goals)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


@dataclass
class EloRatings:
    """Mutable Elo table for a league.

    The builder updates this table strictly in kickoff order and only ever
    reads a rating *before* applying the match it is describing, which is what
    keeps the resulting features free of the match being predicted.
    """

    initial_rating: float = INITIAL_RATING
    k_factor: float = K_FACTOR
    home_advantage: float = HOME_ADVANTAGE
    season_regression: float = SEASON_REGRESSION
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        """Current rating for a club, seeding an unseen club at the mean.

        A club promoted into the league therefore starts as an average side.
        That is a deliberate v1 simplification: the rating self-corrects within
        a few matches, and the README lists a promoted-team indicator as a
        later feature rather than something to smuggle in here.
        """
        return self.ratings.get(team, self.initial_rating)

    def expected(self, home_team: str, away_team: str) -> float:
        """Expected home score for a fixture, from the current ratings."""
        return expected_home_score(
            self.rating(home_team), self.rating(away_team), self.home_advantage
        )

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
        """Apply one completed match to both clubs' ratings.

        Elo is zero-sum: the points the winner gains are exactly those the
        loser gives up, so the table's mean is preserved.
        """
        home_rating = self.rating(home_team)
        away_rating = self.rating(away_team)

        expected = expected_home_score(home_rating, away_rating, self.home_advantage)
        actual = actual_home_score(home_goals, away_goals)
        adjustment = (
            self.k_factor * goal_difference_multiplier(home_goals, away_goals) * (actual - expected)
        )

        self.ratings[home_team] = home_rating + adjustment
        self.ratings[away_team] = away_rating - adjustment

    def apply_season_regression(self) -> None:
        """Pull every rating toward the mean at a season boundary.

        Squads, managers, and form all change over a summer, so May's rating
        overstates what is actually known in August. Regressing keeps Elo from
        carrying stale certainty into a new season.
        """
        if not self.season_regression:
            return
        for team, rating in self.ratings.items():
            self.ratings[team] = self.initial_rating + (rating - self.initial_rating) * (
                1.0 - self.season_regression
            )

    def copy(self) -> "EloRatings":
        """An independent snapshot, for evaluating a fixture without mutating state."""
        return EloRatings(
            initial_rating=self.initial_rating,
            k_factor=self.k_factor,
            home_advantage=self.home_advantage,
            season_regression=self.season_regression,
            ratings=dict(self.ratings),
        )

    def as_dict(self) -> dict[str, float]:
        """The rating table, for persisting alongside a model artifact."""
        return dict(self.ratings)
