import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeSeriesDecomposer:
    """Performs and visualizes time series decomposition."""

    def __init__(self, time_series: pd.Series):
        """Initialize with the time series data.

        Args:
            time_series (pd.Series): A pandas Series with a DatetimeIndex.
        """
        if not isinstance(time_series.index, pd.DatetimeIndex):
            raise ValueError("Input Series must have a DatetimeIndex.")
        self.time_series = time_series.sort_index()
        self.decomposition_result = None
        self.model_type = None
        self.period = None

    def decompose(self, model='additive', period=12):
        """Performs seasonal decomposition.

        Args:
            model (str): Type of decomposition ('additive' or 'multiplicative').
            period (int): The period of the seasonality.

        Returns:
            bool: True if decomposition was successful, False otherwise.
        """
        self.model_type = model
        self.period = period
        min_required_periods = 2

        if len(self.time_series) < self.period * min_required_periods:
            logging.warning(f"Time series too short for decomposition with period {self.period}. "
                            f"Need at least {self.period * min_required_periods} observations, found {len(self.time_series)}.")
            self.decomposition_result = None
            return False

        try:
            # Ensure the Series has a frequency set if possible, otherwise statsmodels might warn/error
            # common_freq = pd.infer_freq(self.time_series.index)
            # if common_freq:
            #     self.time_series = self.time_series.asfreq(common_freq)
            # else:
            #     logging.warning("Could not infer frequency for the time series. Decomposition might be less accurate.")
            
            # Fill missing values if any - necessary for seasonal_decompose
            ts_filled = self.time_series.interpolate(method='time')
            if ts_filled.isnull().any():
                 ts_filled = ts_filled.fillna(method='bfill').fillna(method='ffill') # Handle edges
            
            # Check again after filling, if still null, cannot proceed
            if ts_filled.isnull().any():
                 logging.error("Time series contains NaNs even after interpolation and fillna. Cannot decompose.")
                 self.decomposition_result = None
                 return False

            self.decomposition_result = seasonal_decompose(
                ts_filled, 
                model=self.model_type, 
                period=self.period,
                extrapolate_trend='freq' # Helps avoid NaNs at ends of trend
            )
            logging.info(f"Time series decomposition successful using model='{model}', period={period}.")
            return True
        except Exception as e:
            logging.error(f"Error during time series decomposition: {e}")
            self.decomposition_result = None
            return False

    def plot_decomposition(self, title="Time Series Decomposition") -> go.Figure:
        """Generates an interactive plot of the decomposition results.
        
        Args:
            title (str): The title for the plot.

        Returns:
            go.Figure: A Plotly figure object, or an empty figure if decomposition failed.
        """
        if self.decomposition_result is None:
            logging.warning("Decomposition has not been run or failed. Cannot plot.")
            # Return an empty figure with an annotation
            fig = go.Figure()
            fig.update_layout(title=f"{title} (Failed)", 
                              xaxis_visible=False, yaxis_visible=False,
                              annotations=[{
                                  "text": "Decomposition could not be performed.<br>Check logs or data.",
                                  "xref": "paper", "yref": "paper",
                                  "showarrow": False, "font": {"size": 16}
                              }])
            return fig

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.05
        )

        # Observed
        fig.add_trace(go.Scatter(x=self.decomposition_result.observed.index,
                                 y=self.decomposition_result.observed,
                                 mode='lines', name='Observed'), row=1, col=1)

        # Trend
        fig.add_trace(go.Scatter(x=self.decomposition_result.trend.index,
                                 y=self.decomposition_result.trend,
                                 mode='lines', name='Trend'), row=2, col=1)

        # Seasonal
        fig.add_trace(go.Scatter(x=self.decomposition_result.seasonal.index,
                                 y=self.decomposition_result.seasonal,
                                 mode='lines', name='Seasonal'), row=3, col=1)

        # Residual
        fig.add_trace(go.Scatter(x=self.decomposition_result.resid.index,
                                 y=self.decomposition_result.resid,
                                 mode='lines+markers', name='Residual', marker=dict(size=4)), row=4, col=1)

        fig.update_layout(
            title_text=f"{title} (Model: {self.model_type.capitalize()}, Period: {self.period})",
            height=700,
            showlegend=False # Traces are clearly labeled by subplot titles
        )
        
        # Improve hover text
        fig.update_traces(hoverinfo='x+y')

        return fig

# Example Usage (Optional - for testing)
if __name__ == '__main__':
    # Create a dummy time series
    dates = pd.date_range(start='2020-01-01', periods=60, freq='M')
    data = (
        100 
        + 5 * (dates.year - 2020) # Trend
        + 10 * pd.Series(data=[1, 2, 3, 2, 1, 0, -1, -2, -3, -2, -1, 0] * 5, index=dates) # Seasonal
        + pd.np.random.randn(60) * 5 # Noise
    )
    ts = pd.Series(data, index=dates)
    ts.iloc[10] = None # Add a missing value
    ts.iloc[50] = None # Add another

    print("Original Time Series:")
    print(ts.head())

    decomposer = TimeSeriesDecomposer(ts)
    
    # Test decomposition
    success = decomposer.decompose(model='additive', period=12)
    
    if success:
        print("\nDecomposition Result (Trend Head):")
        print(decomposer.decomposition_result.trend.head())
        
        # Test plotting
        fig = decomposer.plot_decomposition(title="Example Decomposition")
        # To show the plot in a non-Streamlit environment, you might need:
        # fig.show() 
        print("\nPlot generated successfully.")
    else:
        print("\nDecomposition failed.")

    # Test short series
    short_ts = ts.head(20)
    print("\nTesting short series:")
    short_decomposer = TimeSeriesDecomposer(short_ts)
    success_short = short_decomposer.decompose()
    if not success_short:
        print("Decomposition correctly failed for short series.")
        fig_short = short_decomposer.plot_decomposition(title="Short Series Decomposition")
        # fig_short.show()

