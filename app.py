"""
NBA Optimal Team Selection using Multilayer Perceptron (MLP)
AIT-204 Topic 2 Assignment

This Streamlit application demonstrates:
1. MLP architecture with forward and backward propagation
2. Training on NBA player statistics
3. Optimal team selection based on neural network predictions

Author: AIT-204 Student
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the mlp module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlp.neural_network import MultilayerPerceptron, TeamSelectionMLP, ActivationFunctions
from mlp.data_processing import NBADataProcessor, TeamBuilder, load_sample_data, load_nba_data


# Page configuration
st.set_page_config(
    page_title="NBA Team Selection - MLP Neural Network",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def create_network_visualization(layer_dims):
    """Create a visual representation of the neural network architecture."""
    fig = go.Figure()

    max_neurons = max(layer_dims)
    x_spacing = 1
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for layer_idx, n_neurons in enumerate(layer_dims):
        x = layer_idx * x_spacing
        y_positions = np.linspace(0, max_neurons, n_neurons + 2)[1:-1]

        # Draw neurons
        for y in y_positions:
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=30,
                    color=colors[layer_idx % len(colors)],
                    line=dict(width=2, color='black')
                ),
                showlegend=False,
                hoverinfo='text',
                hovertext=f'Layer {layer_idx}, Neuron'
            ))

        # Draw connections to next layer
        if layer_idx < len(layer_dims) - 1:
            next_n_neurons = layer_dims[layer_idx + 1]
            next_y_positions = np.linspace(0, max_neurons, next_n_neurons + 2)[1:-1]

            for y1 in y_positions:
                for y2 in next_y_positions:
                    fig.add_trace(go.Scatter(
                        x=[x, x + x_spacing],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(color='rgba(100,100,100,0.2)', width=0.5),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

    # Add layer labels
    layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, len(layer_dims) - 1)] + ['Output']
    for idx, name in enumerate(layer_names):
        fig.add_annotation(
            x=idx * x_spacing,
            y=-1,
            text=f"{name}<br>({layer_dims[idx]} neurons)",
            showarrow=False,
            font=dict(size=12)
        )

    fig.update_layout(
        title="Neural Network Architecture",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor='white'
    )

    return fig


def plot_training_history(cost_history, accuracy_history):
    """Create training history plots."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cost Function', 'Training Accuracy')
    )

    # Cost plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cost_history))),
            y=cost_history,
            mode='lines',
            name='Cost',
            line=dict(color='#FF6B6B', width=2)
        ),
        row=1, col=1
    )

    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(accuracy_history))),
            y=accuracy_history,
            mode='lines',
            name='Accuracy',
            line=dict(color='#4ECDC4', width=2)
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Cost", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    return fig


def plot_player_scores(df, top_n=20):
    """Create player score visualization."""
    top_players = df.nlargest(top_n, 'mlp_score')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_players['player_name'],
        y=top_players['mlp_score'],
        marker_color='#1E88E5',
        text=top_players['mlp_score'].round(3),
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Top {top_n} Players by MLP Score",
        xaxis_title="Player",
        yaxis_title="MLP Score",
        xaxis_tickangle=45,
        height=500
    )

    return fig


def plot_team_radar(team_df):
    """Create radar chart for team analysis."""
    categories = ['Points', 'Rebounds', 'Assists', 'Net Rating', 'MLP Score']

    # Normalize values for radar chart
    values = []
    for col in ['pts', 'reb', 'ast', 'net_rating', 'mlp_score']:
        if col in team_df.columns:
            val = team_df[col].mean()
            # Normalize to 0-1 scale
            if col == 'net_rating':
                val = (val + 20) / 40  # Assuming net rating range -20 to 20
            elif col == 'pts':
                val = val / 30
            elif col == 'reb':
                val = val / 15
            elif col == 'ast':
                val = val / 10
            values.append(min(1, max(0, val)))
        else:
            values.append(0)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(color='#1E88E5', width=2),
        name='Team Average'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Team Performance Radar Chart",
        height=400
    )

    return fig


def main():
    """Main application function."""

    st.markdown('<h1 class="main-header">NBA Optimal Team Selection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Using Multilayer Perceptron Neural Network</p>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["NBA Data (Real)", "Sample Data (Demo)", "Upload CSV"]
    )

    # Player pool settings
    st.sidebar.subheader("Player Pool Selection")
    n_players = st.sidebar.slider("Number of players in pool:", 50, 200, 100)

    # Load data based on selection - get parameters first for change detection
    if data_source == "NBA Data (Real)":
        # Settings for real NBA data
        start_year = st.sidebar.slider("Starting season year:", 2010, 2022, 2018)
        min_games = st.sidebar.slider("Minimum games played:", 10, 50, 20)
    else:
        start_year = None
        min_games = None

    # Create a data configuration key to detect any data-related changes
    data_config = {
        'data_source': data_source,
        'n_players': n_players,
        'start_year': start_year,
        'min_games': min_games
    }

    # Detect data configuration changes and clear stale session state
    if 'data_config' not in st.session_state:
        st.session_state['data_config'] = data_config
    elif st.session_state['data_config'] != data_config:
        # Data configuration changed - clear model-related session state
        keys_to_clear = ['mlp', 'cost_history', 'accuracy_history', 'processor',
                         'df_processed', 'X', 'Y', 'n_features']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['data_config'] = data_config
        st.toast("Data configuration changed. Please retrain the model.", icon="⚠️")

    # Load data based on selection
    if data_source == "NBA Data (Real)":
        df = load_nba_data(
            start_year=start_year,
            n_players=n_players,
            min_games=min_games
        )
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload NBA Players CSV",
            type=['csv']
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file or select another data source")
            df = load_nba_data(n_players=n_players)
    else:
        df = load_sample_data()

    # Network architecture
    st.sidebar.subheader("MLP Architecture")
    n_hidden_layers = st.sidebar.slider("Number of hidden layers:", 1, 5, 3)

    hidden_layer_sizes = []
    for i in range(n_hidden_layers):
        size = st.sidebar.slider(f"Hidden layer {i+1} neurons:", 8, 128, 64 // (i+1))
        hidden_layer_sizes.append(size)

    # Training parameters
    st.sidebar.subheader("Training Parameters")
    learning_rate = st.sidebar.select_slider(
        "Learning rate:",
        options=[0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01
    )
    epochs = st.sidebar.slider("Training epochs:", 100, 2000, 500)
    threshold = st.sidebar.slider("Classification threshold:", 0.3, 0.8, 0.5)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Problem Overview",
        "MLP Architecture",
        "Training",
        "Team Selection",
        "Analysis"
    ])

    # Tab 1: Problem Overview
    with tab1:
        st.markdown('<h2 class="section-header">Problem Statement</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Objective
            Build an optimal NBA team of 5 players from a pool of 100 players
            using a Multilayer Perceptron (MLP) neural network.

            ### Approach
            1. **Data Collection**: NBA player statistics
            2. **Feature Engineering**: Select relevant performance metrics
            3. **MLP Training**: Learn to identify "optimal" players
            4. **Team Selection**: Balance positions and maximize team quality

            ### Player Features Used
            - **Offensive**: Points (PTS), Assists (AST), True Shooting %
            - **Defensive**: Rebounds (REB), Defensive Rating
            - **Efficiency**: Net Rating, Usage Rate, Assist %
            """)

        with col2:
            st.markdown("""
            ### What Makes an "Optimal" Team?
            An optimal team requires:

            - **Balanced Scoring**: Multiple scoring options
            - **Defensive Presence**: Rebounding and rim protection
            - **Playmaking**: Ball handlers and distributors
            - **Position Diversity**: Guards, Forwards, and Centers
            - **Efficiency**: High-percentage shooters

            The MLP learns to identify players who contribute to
            these team-building criteria based on historical data.
            """)

        st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            if 'pts' in df.columns:
                st.metric("Avg Points", f"{df['pts'].mean():.1f}")
        with col4:
            if 'gp' in df.columns:
                st.metric("Avg Games", f"{df['gp'].mean():.1f}")

        st.dataframe(df.head(10), use_container_width=True)

    # Tab 2: MLP Architecture
    with tab2:
        st.markdown('<h2 class="section-header">Neural Network Architecture</h2>', unsafe_allow_html=True)

        # Initialize data processor to get feature count
        processor = NBADataProcessor()
        processor.feature_names = ['pts', 'reb', 'ast', 'net_rating', 'oreb_pct',
                                   'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']
        available_features = [f for f in processor.feature_names if f in df.columns]
        n_features = len(available_features)

        # Build layer dimensions
        layer_dims = [n_features] + hidden_layer_sizes + [1]

        # Network visualization
        fig = create_network_visualization(layer_dims)
        st.plotly_chart(fig, use_container_width=True)

        # Architecture details
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Layer Configuration")
            st.markdown(f"""
            | Layer | Neurons | Activation |
            |-------|---------|------------|
            | Input | {n_features} | - |
            """)

            for i, size in enumerate(hidden_layer_sizes):
                st.markdown(f"| Hidden {i+1} | {size} | ReLU |")

            st.markdown("| Output | 1 | Sigmoid |")

        with col2:
            st.markdown("### Key Components")
            st.markdown("""
            **Forward Propagation:**
            ```
            Z[l] = W[l] · A[l-1] + b[l]
            A[l] = g(Z[l])
            ```

            **Backpropagation:**
            ```
            dZ[l] = dA[l] * g'(Z[l])
            dW[l] = (1/m) * dZ[l] · A[l-1].T
            db[l] = (1/m) * Σ dZ[l]
            ```

            **Weight Update (Gradient Descent):**
            ```
            W[l] = W[l] - α * dW[l]
            b[l] = b[l] - α * db[l]
            ```
            """)

        # Calculate total parameters
        total_params = 0
        for i in range(len(layer_dims) - 1):
            weights = layer_dims[i] * layer_dims[i+1]
            biases = layer_dims[i+1]
            total_params += weights + biases

        st.info(f"Total trainable parameters: **{total_params:,}**")

    # Tab 3: Training
    with tab3:
        st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)

        if st.button("Train MLP", type="primary"):
            with st.spinner("Training neural network..."):
                # Prepare data
                processor = NBADataProcessor()
                df_processed = processor.engineer_features(df.copy(), 'extended')

                # Get available features
                available_features = [f for f in processor.feature_names if f in df_processed.columns]
                n_features = len(available_features)

                # Prepare training data
                X, Y = processor.prepare_training_data(df_processed, threshold=threshold)

                # Store feature count for later use
                st.session_state['n_features'] = n_features
                st.session_state['processor'] = processor
                st.session_state['df_processed'] = df_processed
                st.session_state['X'] = X
                st.session_state['Y'] = Y

                # Create and train MLP
                mlp = TeamSelectionMLP(
                    n_features=n_features,
                    hidden_layers=hidden_layer_sizes,
                    learning_rate=learning_rate,
                    random_seed=42
                )

                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                cost_history = []
                accuracy_history = []

                # Manual training loop for progress tracking
                for epoch in range(epochs):
                    # Forward propagation
                    AL = mlp.forward_propagation(X)

                    # Compute cost
                    cost = mlp.compute_cost(AL, Y)
                    cost_history.append(cost)

                    # Compute accuracy
                    predictions = (AL > 0.5).astype(int)
                    accuracy = np.mean(predictions == Y)
                    accuracy_history.append(accuracy)

                    # Backward propagation
                    gradients = mlp.backward_propagation(Y)

                    # Update parameters
                    mlp.update_parameters(gradients)

                    # Update progress
                    if epoch % 10 == 0:
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f"Epoch {epoch}: Cost = {cost:.6f}, Accuracy = {accuracy:.4f}")

                progress_bar.progress(1.0)
                status_text.text(f"Training complete! Final Cost: {cost_history[-1]:.6f}, Final Accuracy: {accuracy_history[-1]:.4f}")

                # Store trained model and history
                st.session_state['mlp'] = mlp
                st.session_state['cost_history'] = cost_history
                st.session_state['accuracy_history'] = accuracy_history

        # Display training results if available
        if 'cost_history' in st.session_state:
            st.markdown("### Training Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Cost", f"{st.session_state['cost_history'][-1]:.6f}")
            with col2:
                st.metric("Final Accuracy", f"{st.session_state['accuracy_history'][-1]:.2%}")
            with col3:
                st.metric("Epochs Completed", len(st.session_state['cost_history']))

            # Training history plot
            fig = plot_training_history(
                st.session_state['cost_history'],
                st.session_state['accuracy_history']
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Training Interpretation")
            st.markdown("""
            **Cost Function (Binary Cross-Entropy):**
            - Measures the difference between predicted probabilities and actual labels
            - Lower cost indicates better model fit
            - Should decrease over training epochs

            **Accuracy:**
            - Percentage of correct player classifications
            - High accuracy indicates the model successfully learned player quality patterns
            """)

    # Tab 4: Team Selection
    with tab4:
        st.markdown('<h2 class="section-header">Optimal Team Selection</h2>', unsafe_allow_html=True)

        if 'mlp' not in st.session_state:
            st.warning("Please train the model first in the 'Training' tab.")
        else:
            mlp = st.session_state['mlp']
            processor = st.session_state['processor']
            df_processed = st.session_state['df_processed']
            X = st.session_state['X']

            # Get predictions
            scores = mlp.predict_proba(X).flatten()
            df_processed = df_processed.copy()
            df_processed['mlp_score'] = scores

            # Build team
            team_builder = TeamBuilder(team_size=5)
            optimal_team = team_builder.select_optimal_team(
                df_processed,
                scores,
                balance_positions=True
            )

            st.markdown("### Selected Optimal Team")

            # Display team
            cols = st.columns(5)
            display_cols = ['player_name', 'pts', 'reb', 'ast', 'net_rating', 'mlp_score']
            available_display = [c for c in display_cols if c in optimal_team.columns]

            for i, (idx, player) in enumerate(optimal_team.iterrows()):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1E88E5, #42A5F5);
                                padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                        <h4 style="margin: 0;">{player.get('player_name', f'Player {i+1}')}</h4>
                        <p style="margin: 0.5rem 0;">Position: {player.get('position_category', 'N/A')}</p>
                        <p style="margin: 0;">Score: {player.get('mlp_score', 0):.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("### Team Statistics")
            team_display = optimal_team[available_display].copy()
            st.dataframe(team_display, use_container_width=True)

            # Team evaluation
            evaluation = team_builder.evaluate_team_balance(optimal_team)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Position Distribution")
                if evaluation['position_distribution']:
                    fig = px.pie(
                        values=list(evaluation['position_distribution'].values()),
                        names=list(evaluation['position_distribution'].keys()),
                        title="Team Position Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Team Performance Profile")
                fig = plot_team_radar(optimal_team)
                st.plotly_chart(fig, use_container_width=True)

            # Top players chart
            st.markdown("### All Player Rankings")
            fig = plot_player_scores(df_processed, top_n=20)
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Analysis
    with tab5:
        st.markdown('<h2 class="section-header">Analysis and Findings</h2>', unsafe_allow_html=True)

        st.markdown("""
        ### Algorithm Overview

        The Multilayer Perceptron (MLP) approach to team selection works as follows:

        1. **Data Preprocessing**
           - Normalize player statistics to standard scale
           - Create composite "quality" scores for training labels
           - Handle missing values and outliers

        2. **Forward Propagation**
           - Input layer receives player statistics
           - Hidden layers extract hierarchical features
           - Output layer produces suitability probability

        3. **Backpropagation**
           - Calculate error between predictions and labels
           - Propagate gradients backward through network
           - Update weights using gradient descent

        4. **Team Selection**
           - Rank all players by MLP output score
           - Apply position constraints for team balance
           - Select top 5 players meeting criteria

        ### Key Findings

        The MLP successfully learns to identify valuable players based on:
        - **Offensive Production**: Points and assists are strong indicators
        - **Efficiency Metrics**: Net rating shows overall impact
        - **Versatility**: Players with balanced stats score higher

        ### Limitations

        - Model trained on limited historical data
        - Doesn't account for player chemistry
        - Static evaluation doesn't consider game situations

        ### Future Improvements

        - Include advanced metrics (RAPTOR, BPM, VORP)
        - Add injury history and durability factors
        - Implement ensemble methods for robustness
        """)

        if 'mlp' in st.session_state:
            st.markdown("### Model Architecture Summary")
            st.code(st.session_state['mlp'].get_architecture_summary())

        st.markdown("""
        ### References

        1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations
           by back-propagating errors. *Nature*, 323(6088), 533-536.

        2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
           Surpassing human-level performance on ImageNet classification. *ICCV*.

        3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

        4. NBA Stats: https://www.nba.com/stats

        5. Kaggle NBA Players Dataset: https://www.kaggle.com/datasets/justinas/nba-players-data
        """)


if __name__ == "__main__":
    main()
