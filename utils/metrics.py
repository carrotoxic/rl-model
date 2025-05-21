import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def calculate_return(portfolio_values: np.ndarray) -> float:
    """
    Calculate total return.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Total return
    """
    return (portfolio_values[-1] / portfolio_values[0]) - 1


def calculate_annualized_return(portfolio_values: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        portfolio_values: Array of portfolio values over time
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Annualized return
    """
    total_return = calculate_return(portfolio_values)
    n_days = len(portfolio_values) - 1
    return (1 + total_return) ** (trading_days_per_year / n_days) - 1


def calculate_sharpe_ratio(portfolio_values: np.ndarray, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        portfolio_values: Array of portfolio values over time
        risk_free_rate: Risk-free rate (annualized)
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Sharpe ratio
    """
    # Calculate daily returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    
    # Calculate excess returns
    excess_returns = daily_returns - daily_rf
    
    # Calculate Sharpe ratio (annualized)
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days_per_year)
    
    return sharpe


def calculate_sortino_ratio(portfolio_values: np.ndarray, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        portfolio_values: Array of portfolio values over time
        risk_free_rate: Risk-free rate (annualized)
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Sortino ratio
    """
    # Calculate daily returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    
    # Calculate excess returns
    excess_returns = daily_returns - daily_rf
    
    # Calculate negative returns (downside)
    negative_returns = excess_returns[excess_returns < 0]
    
    # If no negative returns, return a high value
    if len(negative_returns) == 0:
        return np.inf
    
    # Calculate Sortino ratio (annualized)
    sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(trading_days_per_year)
    
    return sortino


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Maximum drawdown
    """
    # Calculate the maximum drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return max_drawdown


def calculate_calmar_ratio(portfolio_values: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        portfolio_values: Array of portfolio values over time
        trading_days_per_year: Number of trading days per year
        
    Returns:
        Calmar ratio
    """
    # Calculate annualized return
    ann_return = calculate_annualized_return(portfolio_values, trading_days_per_year)
    
    # Calculate maximum drawdown
    max_dd = calculate_max_drawdown(portfolio_values)
    
    # If no drawdown, return a high value
    if max_dd == 0:
        return np.inf
    
    # Calculate Calmar ratio
    calmar = ann_return / abs(max_dd)
    
    return calmar


def evaluate_portfolio(portfolio_values: np.ndarray, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Evaluate portfolio performance.
    
    Args:
        portfolio_values: Array of portfolio values over time
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {
        'total_return': calculate_return(portfolio_values),
        'annualized_return': calculate_annualized_return(portfolio_values),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(portfolio_values, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'calmar_ratio': calculate_calmar_ratio(portfolio_values)
    }
    
    return metrics


def plot_portfolio_performance(
    portfolio_values: np.ndarray,
    benchmark_values: np.ndarray = None,
    title: str = "Portfolio Performance",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot portfolio performance over time.
    
    Args:
        portfolio_values: Array of portfolio values over time
        benchmark_values: Array of benchmark values over time (optional)
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot portfolio values
    plt.plot(portfolio_values, label="Portfolio", linewidth=2)
    
    # Plot benchmark values if provided
    if benchmark_values is not None:
        plt.plot(benchmark_values, label="Benchmark", linewidth=2, alpha=0.7)
    
    # Add metrics to the plot
    metrics = evaluate_portfolio(portfolio_values)
    metrics_text = (
        f"Return: {metrics['total_return']:.2%}\n"
        f"Sharpe: {metrics['sharpe_ratio']:.2f}\n"
        f"Max DD: {metrics['max_drawdown']:.2%}\n"
        f"Calmar: {metrics['calmar_ratio']:.2f}"
    )
    
    plt.text(
        0.02, 0.98, metrics_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    # Add legend, grid, and labels
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value")
    
    plt.tight_layout()
    plt.show()


def plot_drawdowns(portfolio_values: np.ndarray, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot drawdowns over time.
    
    Args:
        portfolio_values: Array of portfolio values over time
        figsize: Figure size
    """
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    
    plt.figure(figsize=figsize)
    plt.plot(drawdown, color='red', linewidth=2)
    plt.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title("Portfolio Drawdowns")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown")
    
    # Add maximum drawdown to the plot
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    plt.axhline(y=max_dd, color='black', linestyle='--', alpha=0.5)
    plt.text(
        max_dd_idx, max_dd, f" Max Drawdown: {max_dd:.2%}",
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    plt.show()


def print_performance_report(portfolio_values: np.ndarray, benchmark_values: np.ndarray = None) -> None:
    """
    Print a comprehensive performance report.
    
    Args:
        portfolio_values: Array of portfolio values over time
        benchmark_values: Array of benchmark values over time (optional)
    """
    metrics = evaluate_portfolio(portfolio_values)
    
    print("=" * 50)
    print("PORTFOLIO PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
    
    if benchmark_values is not None:
        benchmark_metrics = evaluate_portfolio(benchmark_values)
        
        print("\n" + "=" * 50)
        print("BENCHMARK COMPARISON")
        print("=" * 50)
        print(f"Portfolio Return: {metrics['total_return']:.2%}")
        print(f"Benchmark Return: {benchmark_metrics['total_return']:.2%}")
        print(f"Alpha: {metrics['total_return'] - benchmark_metrics['total_return']:.2%}")
        
        print(f"Portfolio Sharpe: {metrics['sharpe_ratio']:.4f}")
        print(f"Benchmark Sharpe: {benchmark_metrics['sharpe_ratio']:.4f}")
        
        print(f"Portfolio Max DD: {metrics['max_drawdown']:.2%}")
        print(f"Benchmark Max DD: {benchmark_metrics['max_drawdown']:.2%}")
    
    print("=" * 50) 