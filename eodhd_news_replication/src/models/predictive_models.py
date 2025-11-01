"""
Predictive Regression Models and VAR Implementation
Includes LASSO feature selection and backtesting
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PredictiveModels:
    """Predictive regression models for return forecasting"""

    def __init__(self, config_path: str = "conf/experiment.yaml"):
        """Initialize predictive models with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Model parameters
        self.forecast_horizon = self.config['predictive_models']['forecast_horizon']
        self.train_window = self.config['predictive_models']['train_window']
        self.fixed_window = self.config['predictive_models']['fixed_window']
        self.test_split = self.config['predictive_models']['test_split']
        self.lasso_cv_folds = self.config['predictive_models']['lasso_cv_folds']
        self.max_selected_topics = self.config['predictive_models']['max_selected_topics']
        self.transaction_cost_bps = self.config['predictive_models']['transaction_cost_bps']

        # VAR parameters
        self.var_max_lags = self.config['var']['max_lags']
        self.bootstrap_reps = self.config['var']['bootstrap_reps']
        self.irf_horizon = self.config['var']['irf_horizon']

    def load_panel_data(self, frequency: str = 'monthly', method: str = 'mean',
                       model_type: str = 'online') -> pd.DataFrame:
        """Load panel data with topics and returns"""
        panel_file = f"data/derived/panel_{frequency}_{method}_{model_type}.parquet"

        if not Path(panel_file).exists():
            # Try batch model if online not available
            panel_file = f"data/derived/panel_{frequency}_{method}_batch.parquet"

        if not Path(panel_file).exists():
            raise FileNotFoundError(f"Panel data not found: {panel_file}")

        panel = pd.read_parquet(panel_file)

        # Ensure we have returns
        if 'log_return' not in panel.columns:
            raise ValueError("Returns not found in panel data")

        # Drop rows with missing returns
        panel = panel.dropna(subset=['log_return'])

        logger.info(f"Loaded panel with {len(panel)} observations from {panel_file}")

        return panel

    def prepare_features_targets(self, panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features (topics) and targets (future returns)"""
        # Get topic columns
        topic_cols = [col for col in panel.columns if col.startswith('topic_') and not col.endswith('_smoothed')]

        # Features: current topics
        X = panel[topic_cols].copy()

        # Target: next period return
        y = panel['log_return'].shift(-self.forecast_horizon)

        # Remove last rows where we don't have future returns
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]

        logger.info(f"Prepared {len(X)} observations with {len(topic_cols)} topic features")

        return X, y

    def train_lasso_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train LASSO model with cross-validation"""
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train LASSO with CV
        lasso = LassoCV(cv=self.lasso_cv_folds, random_state=42, max_iter=5000)
        lasso.fit(X_train_scaled, y_train)

        # Get selected features
        selected_features = X_train.columns[lasso.coef_ != 0].tolist()

        # Limit to max topics
        if len(selected_features) > self.max_selected_topics:
            # Sort by absolute coefficient value
            coef_abs = np.abs(lasso.coef_[lasso.coef_ != 0])
            top_indices = np.argsort(coef_abs)[-self.max_selected_topics:]
            selected_features = [selected_features[i] for i in top_indices]

        # Predictions
        y_pred_train = lasso.predict(X_train_scaled)
        y_pred_val = lasso.predict(X_val_scaled)

        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)

        # Benchmark: historical mean
        benchmark_pred = np.full_like(y_val, y_train.mean())
        benchmark_r2 = r2_score(y_val, benchmark_pred)

        results = {
            'model': lasso,
            'scaler': scaler,
            'selected_features': selected_features,
            'coefficients': dict(zip(selected_features,
                                   lasso.coef_[lasso.coef_ != 0][:len(selected_features)])),
            'alpha': lasso.alpha_,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'benchmark_r2': benchmark_r2,
            'oos_r2_vs_benchmark': val_r2 - benchmark_r2
        }

        logger.info(f"LASSO selected {len(selected_features)} features, "
                   f"Val R2: {val_r2:.4f}, Benchmark R2: {benchmark_r2:.4f}")

        return results

    def rolling_window_backtest(self, panel: pd.DataFrame, window_type: str = 'expanding') -> pd.DataFrame:
        """Perform rolling window out-of-sample prediction"""
        X, y = self.prepare_features_targets(panel)

        # Determine train/test split
        n_obs = len(X)
        n_test = int(n_obs * self.test_split)
        n_train = n_obs - n_test

        logger.info(f"Backtesting with {n_train} train and {n_test} test observations")

        predictions = []

        for t in range(n_train, n_obs):
            # Define training window
            if window_type == 'expanding':
                train_start = 0
            else:  # fixed window
                train_start = max(0, t - self.fixed_window)

            train_end = t

            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[t:t+1]
            y_test = y.iloc[t:t+1]

            # Skip if not enough training data
            if len(X_train) < 60:
                continue

            # Train model (simplified - using only LASSO here)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Use simple LASSO (no CV for speed in backtest)
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.01, random_state=42, max_iter=5000)
            lasso.fit(X_train_scaled, y_train)

            # Predict
            y_pred = lasso.predict(X_test_scaled)[0]

            # Store results
            predictions.append({
                'date': X.index[t],
                'actual': y_test.iloc[0],
                'predicted': y_pred,
                'n_selected': (lasso.coef_ != 0).sum()
            })

            if len(predictions) % 10 == 0:
                logger.info(f"  Processed {len(predictions)}/{n_test} test observations")

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        predictions_df.set_index('date', inplace=True)

        # Calculate cumulative statistics
        predictions_df['squared_error'] = (predictions_df['actual'] - predictions_df['predicted']) ** 2
        predictions_df['benchmark_pred'] = predictions_df['actual'].expanding().mean().shift(1)
        predictions_df['benchmark_squared_error'] = (predictions_df['actual'] - predictions_df['benchmark_pred']) ** 2

        # Out-of-sample R²
        oos_r2 = 1 - (predictions_df['squared_error'].sum() /
                     ((predictions_df['actual'] - predictions_df['actual'].mean()) ** 2).sum())

        benchmark_r2 = 1 - (predictions_df['benchmark_squared_error'].sum() /
                           ((predictions_df['actual'] - predictions_df['actual'].mean()) ** 2).sum())

        logger.info(f"OOS R²: {oos_r2:.4f}, Benchmark R²: {benchmark_r2:.4f}")

        return predictions_df

    def calculate_trading_performance(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate trading strategy performance"""
        # Trading signal: long if predicted return > 0
        predictions_df['signal'] = np.where(predictions_df['predicted'] > 0, 1, 0)

        # Strategy returns
        predictions_df['strategy_return'] = predictions_df['signal'] * predictions_df['actual']

        # Transaction costs
        predictions_df['trade'] = predictions_df['signal'].diff().abs()
        predictions_df['transaction_cost'] = predictions_df['trade'] * self.transaction_cost_bps / 10000

        # Net returns
        predictions_df['net_return'] = predictions_df['strategy_return'] - predictions_df['transaction_cost']

        # Cumulative returns
        predictions_df['cum_strategy'] = (1 + predictions_df['net_return']).cumprod()
        predictions_df['cum_market'] = (1 + predictions_df['actual']).cumprod()

        # Performance metrics
        n_periods = len(predictions_df)
        annualization_factor = 12  # For monthly data

        strategy_return = predictions_df['net_return'].mean() * annualization_factor
        strategy_vol = predictions_df['net_return'].std() * np.sqrt(annualization_factor)
        sharpe_ratio = strategy_return / strategy_vol if strategy_vol > 0 else 0

        market_return = predictions_df['actual'].mean() * annualization_factor
        market_vol = predictions_df['actual'].std() * np.sqrt(annualization_factor)
        market_sharpe = market_return / market_vol if market_vol > 0 else 0

        # Hit rate
        hit_rate = ((predictions_df['predicted'] > 0) == (predictions_df['actual'] > 0)).mean()

        # Maximum drawdown
        cum_returns = predictions_df['cum_strategy']
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        performance = {
            'strategy_annual_return': strategy_return,
            'strategy_annual_vol': strategy_vol,
            'strategy_sharpe': sharpe_ratio,
            'market_annual_return': market_return,
            'market_annual_vol': market_vol,
            'market_sharpe': market_sharpe,
            'hit_rate': hit_rate,
            'max_drawdown': max_drawdown,
            'n_trades': predictions_df['trade'].sum(),
            'total_transaction_costs': predictions_df['transaction_cost'].sum()
        }

        logger.info(f"Strategy Sharpe: {sharpe_ratio:.3f}, Hit Rate: {hit_rate:.3f}")

        return performance

    def train_var_model(self, panel: pd.DataFrame, selected_topics: List[str] = None) -> Dict:
        """Train Vector Autoregression model"""
        # Select variables
        if selected_topics is None:
            # Use top topics by variance
            topic_cols = [col for col in panel.columns if col.startswith('topic_')]
            topic_vars = panel[topic_cols].var().nlargest(5).index.tolist()
        else:
            topic_vars = selected_topics

        # Create VAR dataset
        var_data = panel[topic_vars + ['log_return']].dropna()

        # Standardize
        scaler = StandardScaler()
        var_data_scaled = pd.DataFrame(
            scaler.fit_transform(var_data),
            index=var_data.index,
            columns=var_data.columns
        )

        logger.info(f"Training VAR with {len(topic_vars)} topics and returns")

        # Fit VAR
        model = VAR(var_data_scaled)

        # Select lag order
        lag_results = model.select_order(maxlags=self.var_max_lags)
        optimal_lag = lag_results.aic

        logger.info(f"Optimal lag order by AIC: {optimal_lag}")

        # Fit with optimal lag
        var_fitted = model.fit(optimal_lag)

        # Calculate impulse responses
        irf = var_fitted.irf(self.irf_horizon)

        # Variance decomposition
        fevd = var_fitted.fevd(self.irf_horizon)

        # Granger causality tests
        granger_results = {}
        for topic in topic_vars:
            try:
                test = grangercausalitytests(
                    var_data_scaled[['log_return', topic]],
                    maxlag=optimal_lag,
                    verbose=False
                )
                # Get p-value for optimal lag
                granger_results[topic] = test[optimal_lag][0]['ssr_ftest'][1]
            except:
                granger_results[topic] = 1.0

        results = {
            'model': var_fitted,
            'scaler': scaler,
            'variables': topic_vars + ['log_return'],
            'optimal_lag': optimal_lag,
            'aic': lag_results.aic,
            'bic': lag_results.bic,
            'irf': irf,
            'fevd': fevd,
            'granger_causality': granger_results
        }

        return results

    def plot_impulse_responses(self, var_results: Dict, output_dir: str = "results/irf_plots"):
        """Plot impulse response functions"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        irf = var_results['irf']
        variables = var_results['variables']

        # Plot IRFs for shocks to each variable
        for i, shock_var in enumerate(variables):
            fig, axes = plt.subplots(len(variables), 1, figsize=(10, 2*len(variables)))

            if len(variables) == 1:
                axes = [axes]

            for j, response_var in enumerate(variables):
                irf_values = irf.irfs[:, j, i]
                axes[j].plot(irf_values, linewidth=2)
                axes[j].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                axes[j].set_ylabel(response_var)
                axes[j].grid(True, alpha=0.3)

                if j == 0:
                    axes[j].set_title(f'Impulse Response to {shock_var} Shock')
                if j == len(variables) - 1:
                    axes[j].set_xlabel('Periods')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/irf_{shock_var}.png", dpi=100)
            plt.close()

        logger.info(f"Saved IRF plots to {output_dir}")


def run_predictive_models_pipeline() -> Dict:
    """Complete predictive modeling pipeline"""
    logger.info("Starting predictive models pipeline")

    # Initialize models
    models = PredictiveModels()

    # Load panel data
    panel = models.load_panel_data(frequency='monthly', method='mean', model_type='online')

    # Run rolling window backtest
    logger.info("Running rolling window backtest...")
    predictions = models.rolling_window_backtest(panel, window_type='expanding')

    # Save predictions
    predictions.to_parquet("results/predictions_oos.parquet")
    predictions.to_csv("results/predictions_oos.csv")

    # Calculate trading performance
    performance = models.calculate_trading_performance(predictions)

    # Save performance metrics
    with open("results/trading_performance.json", 'w') as f:
        json.dump(performance, f, indent=2)

    logger.info(f"Trading performance: {performance}")

    # Train VAR model
    logger.info("Training VAR model...")
    var_results = models.train_var_model(panel)

    # Save VAR results
    var_summary = {
        'optimal_lag': int(var_results['optimal_lag']),
        'aic': float(var_results['aic']),
        'bic': float(var_results['bic']),
        'granger_causality': var_results['granger_causality']
    }

    with open("results/var_summary.json", 'w') as f:
        json.dump(var_summary, f, indent=2)

    # Plot impulse responses
    models.plot_impulse_responses(var_results)

    # Create equity curve plot
    plt.figure(figsize=(12, 6))
    predictions['cum_strategy'].plot(label='Topic Strategy', linewidth=2)
    predictions['cum_market'].plot(label='Buy & Hold', linewidth=2, alpha=0.7)
    plt.title('Cumulative Returns: Topic-Based Strategy vs Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/equity_curve.png", dpi=100)
    plt.close()

    logger.info("Predictive models pipeline complete!")

    return {
        'predictions': predictions,
        'performance': performance,
        'var_results': var_results
    }


def main():
    """Main function to run predictive models"""
    logging.basicConfig(level=logging.INFO)
    results = run_predictive_models_pipeline()

    # Print summary
    print("\n=== Predictive Model Results ===")
    print(f"Out-of-sample R²: {results['predictions']['actual'].corr(results['predictions']['predicted'])**2:.4f}")
    print(f"Strategy Sharpe Ratio: {results['performance']['strategy_sharpe']:.3f}")
    print(f"Hit Rate: {results['performance']['hit_rate']:.3f}")
    print(f"Maximum Drawdown: {results['performance']['max_drawdown']:.3f}")


if __name__ == "__main__":
    main()