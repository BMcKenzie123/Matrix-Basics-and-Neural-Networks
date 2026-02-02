"""
NBA Dataset Processing Module

This module handles:
- Loading and preprocessing the NBA players dataset
- Feature engineering for player evaluation
- Team composition analysis
- Data normalization and preparation for MLP training
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass


@dataclass
class TeamComposition:
    """Defines the ideal team composition with position requirements."""
    point_guard: int = 1
    shooting_guard: int = 1
    small_forward: int = 1
    power_forward: int = 1
    center: int = 1

    @property
    def total_players(self) -> int:
        return (self.point_guard + self.shooting_guard +
                self.small_forward + self.power_forward + self.center)


class NBADataProcessor:
    """
    Processes NBA player data for the team selection problem.

    Features used for player evaluation:
    - Offensive: pts, ast, fg_pct, fg3_pct, ft_pct
    - Defensive: reb, oreb_pct, dreb_pct
    - Physical: age, player_height, player_weight
    - Efficiency: net_rating, ts_pct, usg_pct
    """

    POSITION_MAPPING = {
        'G': 'Guard',
        'F': 'Forward',
        'C': 'Center',
        'G-F': 'Guard-Forward',
        'F-G': 'Forward-Guard',
        'F-C': 'Forward-Center',
        'C-F': 'Center-Forward'
    }

    # Features for MLP input
    CORE_FEATURES = [
        'pts', 'reb', 'ast', 'net_rating'
    ]

    EXTENDED_FEATURES = [
        'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct',
        'usg_pct', 'ts_pct', 'ast_pct'
    ]

    FULL_FEATURES = [
        'age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast',
        'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
    ]

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data processor.

        Args:
            data_path: Path to the NBA players CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the NBA players dataset.

        Args:
            data_path: Path to CSV file (overrides constructor path)

        Returns:
            Raw DataFrame
        """
        path = data_path or self.data_path
        if path is None:
            raise ValueError("No data path provided")

        self.raw_data = pd.read_csv(path)
        return self.raw_data

    def filter_by_season_window(self,
                                 df: pd.DataFrame,
                                 start_year: int,
                                 window_size: int = 5) -> pd.DataFrame:
        """
        Filter players within a specified season window.

        Args:
            df: Input DataFrame
            start_year: Starting year (e.g., 2018)
            window_size: Number of years to include

        Returns:
            Filtered DataFrame
        """
        # Extract year from season column (format: "2018-19" or "2018-2019")
        df = df.copy()

        def extract_year(season):
            if pd.isna(season):
                return None
            try:
                return int(str(season).split('-')[0])
            except (ValueError, IndexError):
                return None

        df['season_year'] = df['season'].apply(extract_year)

        end_year = start_year + window_size - 1
        filtered = df[(df['season_year'] >= start_year) &
                     (df['season_year'] <= end_year)]

        return filtered

    def select_player_pool(self,
                           df: pd.DataFrame,
                           n_players: int = 100,
                           min_games: int = 20,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Select a pool of players for team selection.

        Args:
            df: Input DataFrame
            n_players: Number of players to select
            min_games: Minimum games played requirement
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with selected players
        """
        # Filter by minimum games played
        df_filtered = df[df['gp'] >= min_games].copy()

        # Remove duplicates (keep most recent season for each player)
        df_filtered = df_filtered.sort_values('season_year', ascending=False)
        df_filtered = df_filtered.drop_duplicates(subset=['player_name'], keep='first')

        # Select top players by a composite score or random sample
        if len(df_filtered) > n_players:
            # Sample ensuring diversity in positions
            df_sampled = df_filtered.sample(n=min(n_players, len(df_filtered)),
                                           random_state=random_state)
        else:
            df_sampled = df_filtered

        return df_sampled.reset_index(drop=True)

    def engineer_features(self,
                          df: pd.DataFrame,
                          feature_set: str = 'extended') -> pd.DataFrame:
        """
        Engineer features for the MLP.

        Args:
            df: Input DataFrame
            feature_set: 'core', 'extended', or 'full'

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Select feature set
        if feature_set == 'core':
            features = self.CORE_FEATURES
        elif feature_set == 'extended':
            features = self.EXTENDED_FEATURES
        elif feature_set == 'full':
            features = self.FULL_FEATURES
        else:
            features = self.EXTENDED_FEATURES

        self.feature_names = features

        # Handle missing values
        for col in features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def create_player_scores(self,
                              df: pd.DataFrame,
                              weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Create composite player scores for training labels.

        This creates a "quality score" that represents how valuable
        a player is for an optimal team. The score is normalized to [0, 1].

        Args:
            df: DataFrame with player statistics
            weights: Optional custom weights for each statistic

        Returns:
            Array of player scores (labels for training)
        """
        # Default weights for creating player quality scores
        default_weights = {
            'pts': 0.25,        # Scoring is important
            'reb': 0.15,        # Rebounds contribute to team success
            'ast': 0.15,        # Playmaking ability
            'net_rating': 0.20, # Overall impact on team
            'ts_pct': 0.15,     # Shooting efficiency
            'ast_pct': 0.10     # Assist percentage
        }

        weights = weights or default_weights

        # Calculate weighted score
        score = np.zeros(len(df))
        for feature, weight in weights.items():
            if feature in df.columns:
                # Normalize each feature to [0, 1]
                values = df[feature].values
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)
                score += weight * normalized

        # Normalize final score to [0, 1]
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)

        return score

    def prepare_training_data(self,
                               df: pd.DataFrame,
                               feature_set: str = 'extended',
                               threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels for MLP training.

        Args:
            df: Processed DataFrame
            feature_set: Which features to use
            threshold: Score threshold for binary classification

        Returns:
            X: Feature matrix of shape (n_features, n_samples)
            Y: Binary labels of shape (1, n_samples)
        """
        # Get features
        df = self.engineer_features(df, feature_set)
        features = [f for f in self.feature_names if f in df.columns]

        X = df[features].values

        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Normalize features
        X = self.scaler.fit_transform(X)

        # Create labels based on composite score
        scores = self.create_player_scores(df)
        Y = (scores >= threshold).astype(float)

        # Transpose to match MLP expected shape (n_features, n_samples)
        X = X.T
        Y = Y.reshape(1, -1)

        return X, Y

    def get_player_features(self,
                            df: pd.DataFrame,
                            player_name: str) -> Optional[np.ndarray]:
        """
        Get normalized features for a specific player.

        Args:
            df: DataFrame with player data
            player_name: Name of the player

        Returns:
            Feature vector or None if player not found
        """
        player_row = df[df['player_name'] == player_name]
        if len(player_row) == 0:
            return None

        features = [f for f in self.feature_names if f in df.columns]
        X = player_row[features].values

        # Normalize
        X = self.scaler.transform(X)

        return X.T  # Shape: (n_features, 1)


class TeamBuilder:
    """
    Builds optimal teams based on MLP predictions and constraints.

    Considers:
    - Position diversity
    - Player complementarity
    - Overall team balance
    """

    POSITION_CATEGORIES = {
        'Guard': ['G', 'G-F', 'F-G'],
        'Forward': ['F', 'F-G', 'G-F', 'F-C', 'C-F'],
        'Center': ['C', 'C-F', 'F-C']
    }

    def __init__(self, team_size: int = 5):
        """
        Initialize the team builder.

        Args:
            team_size: Number of players on the team
        """
        self.team_size = team_size

    def categorize_position(self, position: str) -> str:
        """Categorize a position into Guard, Forward, or Center."""
        position = str(position).upper() if pd.notna(position) else 'F'
        for category, positions in self.POSITION_CATEGORIES.items():
            if position in positions:
                return category
        return 'Forward'  # Default

    def select_optimal_team(self,
                            df: pd.DataFrame,
                            scores: np.ndarray,
                            balance_positions: bool = True) -> pd.DataFrame:
        """
        Select the optimal team based on MLP scores.

        Args:
            df: DataFrame with player information
            scores: Array of player scores from MLP
            balance_positions: Whether to enforce position balance

        Returns:
            DataFrame with selected team members
        """
        df = df.copy()
        df['mlp_score'] = scores

        if balance_positions:
            # Categorize positions
            if 'player_position' in df.columns:
                df['position_category'] = df['player_position'].apply(
                    self.categorize_position
                )
            else:
                df['position_category'] = 'Forward'

            # Select top players from each position category
            selected = []

            # Ensure at least 1 Guard, 2 Forwards, and 1 Center
            position_targets = {'Guard': 2, 'Forward': 2, 'Center': 1}

            for position, target in position_targets.items():
                position_players = df[df['position_category'] == position]
                top_players = position_players.nlargest(target, 'mlp_score')
                selected.append(top_players)

            team = pd.concat(selected, ignore_index=True)

            # If we need more players, fill from remaining
            if len(team) < self.team_size:
                remaining = df[~df['player_name'].isin(team['player_name'])]
                additional = remaining.nlargest(
                    self.team_size - len(team), 'mlp_score'
                )
                team = pd.concat([team, additional], ignore_index=True)

            # Trim to team size
            team = team.nlargest(self.team_size, 'mlp_score')

        else:
            # Simply select top N players by score
            team = df.nlargest(self.team_size, 'mlp_score')

        return team.reset_index(drop=True)

    def evaluate_team_balance(self, team_df: pd.DataFrame) -> Dict[str, any]:
        """
        Evaluate the balance and quality of a selected team.

        Args:
            team_df: DataFrame with team members

        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            'total_players': len(team_df),
            'average_score': team_df['mlp_score'].mean() if 'mlp_score' in team_df else 0,
            'position_distribution': {},
            'stat_summary': {}
        }

        # Position distribution
        if 'position_category' in team_df.columns:
            evaluation['position_distribution'] = (
                team_df['position_category'].value_counts().to_dict()
            )

        # Calculate team stat averages
        stat_cols = ['pts', 'reb', 'ast', 'net_rating']
        for col in stat_cols:
            if col in team_df.columns:
                evaluation['stat_summary'][col] = {
                    'mean': team_df[col].mean(),
                    'total': team_df[col].sum()
                }

        return evaluation


def load_sample_data(n_players: int = 150,
                     min_games: int = 20,
                     start_year: int = 2020) -> pd.DataFrame:
    """
    Create sample NBA-like data for demonstration.

    Args:
        n_players: Number of players to generate (default: 150)
        min_games: Minimum games played - used to set lower bound for gp (default: 20)
        start_year: Season year for the data (default: 2020)

    Returns:
        DataFrame with synthetic player data
    """
    np.random.seed(42)

    # Format season string based on start_year
    season_str = f'{start_year}-{str(start_year + 1)[-2:]}'

    # Generate synthetic player data
    data = {
        'player_name': [f'Player_{i}' for i in range(n_players)],
        'season': [season_str] * n_players,
        'player_position': np.random.choice(
            ['G', 'F', 'C', 'G-F', 'F-C'], n_players, p=[0.3, 0.3, 0.2, 0.1, 0.1]
        ),
        'age': np.random.randint(20, 38, n_players),
        'player_height': np.random.normal(200, 10, n_players),  # cm
        'player_weight': np.random.normal(100, 15, n_players),  # kg
        'gp': np.random.randint(min_games, 82, n_players),
        'pts': np.random.exponential(12, n_players) + 2,
        'reb': np.random.exponential(4, n_players) + 1,
        'ast': np.random.exponential(3, n_players) + 0.5,
        'net_rating': np.random.normal(0, 8, n_players),
        'oreb_pct': np.random.uniform(0, 15, n_players),
        'dreb_pct': np.random.uniform(5, 25, n_players),
        'usg_pct': np.random.uniform(10, 35, n_players),
        'ts_pct': np.random.uniform(0.45, 0.65, n_players),
        'ast_pct': np.random.uniform(5, 40, n_players)
    }

    df = pd.DataFrame(data)
    df['season_year'] = start_year

    return df


def load_nba_data(data_path: Optional[str] = None,
                  start_year: int = 2018,
                  n_players: int = 100,
                  min_games: int = 20) -> pd.DataFrame:
    """
    Load actual NBA player data from the repository CSV file.

    Args:
        data_path: Path to the CSV file. If None, uses the default path.
        start_year: Starting year for filtering seasons (default: 2018)
        n_players: Number of players to select for the pool (default: 100)
        min_games: Minimum games played requirement (default: 20)

    Returns:
        DataFrame with processed NBA player data
    """
    import os

    # Default to the all_seasons.csv in the repo root
    if data_path is None:
        # Get the directory containing this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the repo root
        repo_root = os.path.dirname(module_dir)
        data_path = os.path.join(repo_root, 'all_seasons.csv')

    if not os.path.exists(data_path):
        print(f"Warning: NBA data file not found at {data_path}. Using sample data.")
        return load_sample_data(n_players=n_players, min_games=min_games, start_year=start_year)

    # Load the CSV
    df = pd.read_csv(data_path)

    # Extract season year from season column (format: "1996-97" or "2020-21")
    def extract_year(season):
        if pd.isna(season):
            return None
        try:
            return int(str(season).split('-')[0])
        except (ValueError, IndexError):
            return None

    df['season_year'] = df['season'].apply(extract_year)

    # Filter by season window (last 5 years from start_year)
    end_year = start_year + 4
    df_filtered = df[(df['season_year'] >= start_year) & (df['season_year'] <= end_year)]

    # Filter by minimum games played
    if 'gp' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['gp'] >= min_games]

    # Remove duplicates - keep the most recent season for each player
    df_filtered = df_filtered.sort_values('season_year', ascending=False)
    df_filtered = df_filtered.drop_duplicates(subset=['player_name'], keep='first')

    # Sample if we have more players than needed
    if len(df_filtered) > n_players:
        df_filtered = df_filtered.sample(n=n_players, random_state=42)

    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)

    print(f"Loaded {len(df_filtered)} players from {start_year} to {end_year} seasons")

    return df_filtered
