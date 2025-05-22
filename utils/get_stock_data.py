'''
Utility functions for downloading, preprocessing, and saving stock data
compatible with the RL trading environment.

Features:
- close price
- high price
- low price
- volume
'''

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


def download_stock_data(
    tickers: List[str], 
    start_date: str, 
    end_date: str, 
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Download historical stock data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval ('1d', '1wk', '1mo')
        
    Returns:
        Dictionary mapping tickers to their historical data
    """
    print(f"Downloading stock data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    stock_data = {}
    
    for i, ticker in enumerate(tickers):
        try:
            # Download data from Yahoo Finance
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                interval=interval,
                progress=False
            )
            
            if len(data) > 0:
                stock_data[ticker] = data
                print(f"  Downloaded {ticker}: {len(data)} days")
            else:
                print(f"  No data found for {ticker}")
                
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
    
    # Report statistics
    successful = len(stock_data)
    print(f"Successfully downloaded data for {successful}/{len(tickers)} tickers")
    
    return stock_data


def preprocess_stock_data(
    stock_data: Dict[str, pd.DataFrame],
    feature_columns: List[str] = ["Close", "High", "Low", "Volume"],
    min_history: int = 252,  # At least one year of data
    required_days: int = None  # Optional specific number of days required
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess downloaded stock data for use in the RL environment.
    
    Args:
        stock_data: Dictionary mapping tickers to their historical data
        feature_columns: List of data columns to extract
        min_history: Minimum history length required
        required_days: Optional specific number of days required
        
    Returns:
        Tuple of (processed_data, valid_tickers)
        processed_data: numpy array of shape [num_stocks, num_days, num_features]
        valid_tickers: list of tickers corresponding to the stocks in processed_data
    """
    print("Preprocessing stock data...")
    
    # Filter tickers with sufficient history
    valid_tickers = []
    valid_data = {}
    
    for ticker, data in stock_data.items():
        if required_days is not None:
            if len(data) == required_days:
                valid_tickers.append(ticker)
                valid_data[ticker] = data
            else:
                print(f"  Dropping {ticker}: incorrect history length ({len(data)} != {required_days} days)")
        elif len(data) >= min_history:
            valid_tickers.append(ticker)
            valid_data[ticker] = data
        else:
            print(f"  Dropping {ticker}: insufficient history ({len(data)} < {min_history} days)")
    
    if not valid_tickers:
        raise ValueError("No valid stocks with sufficient history found")
    
    print(f"  {len(valid_tickers)} stocks have sufficient history")
    
    # Identify common date range
    all_dates = set()
    for data in valid_data.values():
        all_dates.update(data.index)
    
    common_dates = sorted(all_dates)
    
    # Create numpy array for processed data
    num_stocks = len(valid_tickers)
    num_days = len(common_dates)
    num_features = len(feature_columns)
    
    processed_data = np.zeros((num_stocks, num_days, num_features))
    
    # Fill the array with data
    date_to_idx = {date: i for i, date in enumerate(common_dates)}
    
    for stock_idx, ticker in enumerate(valid_tickers):
        data = valid_data[ticker]
        
        for date_idx, date in enumerate(data.index):
            if date in date_to_idx:
                day_idx = date_to_idx[date]
                
                for feat_idx, col in enumerate(feature_columns):
                    if col in data.columns:
                        value = data[col].iloc[date_idx]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]  # Take first value if it's a Series
                        if not pd.isna(value):
                            processed_data[stock_idx, day_idx, feat_idx] = value
    
    # Handle missing values
    # Forward fill
    for stock_idx in range(num_stocks):
        for feat_idx in range(num_features):
            feature_data = processed_data[stock_idx, :, feat_idx]
            mask = feature_data != 0
            
            if np.any(mask):
                # Get first valid value
                first_valid = np.argmax(mask)
                
                # Fill forward
                for i in range(num_days):
                    if i < first_valid:
                        # Before first valid, copy first valid value
                        processed_data[stock_idx, i, feat_idx] = processed_data[stock_idx, first_valid, feat_idx]
                    elif i > 0 and processed_data[stock_idx, i, feat_idx] == 0:
                        # Forward fill
                        processed_data[stock_idx, i, feat_idx] = processed_data[stock_idx, i-1, feat_idx]
    
    print(f"Processed data shape: {processed_data.shape}")
    return processed_data, valid_tickers


def save_processed_data(
    processed_data: np.ndarray,
    tickers: List[str],
    output_dir: str,
    relation_matrix: Optional[np.ndarray] = None
) -> None:
    """
    Save processed stock data to files for use in the RL environment.
    
    Args:
        processed_data: Numpy array of shape [num_stocks, num_days, num_features]
        tickers: List of tickers corresponding to the stocks
        output_dir: Directory to save the files
        relation_matrix: Optional relationship matrix of shape [num_stocks, num_stocks]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save stock data
    stock_data_path = os.path.join("./", output_dir, "stock_data.npy")
    np.save(stock_data_path, processed_data)
    
    # Save tickers
    tickers_path = os.path.join("./", output_dir, "tickers.json")
    with open(tickers_path, 'w') as f:
        json.dump(tickers, f)
    
    # Save relation matrix if provided
    if relation_matrix is not None:
        relation_path = os.path.join("./", output_dir, "relation_matrix.npy")
        np.save(relation_path, relation_matrix)
    
    print(f"Saved processed data to {output_dir}")
    print(f"  - Stock data: {stock_data_path}")
    print(f"  - Tickers: {tickers_path}")
    if relation_matrix is not None:
        print(f"  - Relation matrix: {relation_path}")


def create_correlation_matrix(
    processed_data: np.ndarray,
    tickers: List[str],
    feature_idx: int = 0,  # Use Close price by default
    method: str = "pearson"
) -> np.ndarray:
    """
    Create a correlation matrix based on historical price correlations.
    
    Args:
        processed_data: Numpy array of shape [num_stocks, num_days, num_features]
        tickers: List of tickers corresponding to the stocks
        feature_idx: Index of feature to use for correlation
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix of shape [num_stocks, num_stocks]
    """
    num_stocks = len(tickers)
    
    # Extract price data for all stocks
    price_data = processed_data[:, :, feature_idx]
    
    # Calculate returns (percent change)
    returns = np.zeros_like(price_data)
    for i in range(num_stocks):
        for t in range(1, price_data.shape[1]):
            if price_data[i, t-1] > 0:
                returns[i, t] = (price_data[i, t] / price_data[i, t-1]) - 1
    
    # Create DataFrame for correlation calculation
    returns_df = pd.DataFrame(
        {ticker: returns[i, 1:] for i, ticker in enumerate(tickers)}
    )
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr(method=method).values
    
    # Convert to absolute values and zero out diagonal
    abs_corr = np.abs(corr_matrix)
    np.fill_diagonal(abs_corr, 0)
    
    return abs_corr


def visualize_stock_data(
    processed_data: np.ndarray,
    tickers: List[str],
    output_path: str,
    num_samples: int = 5
) -> None:
    """
    Create visualization of the processed stock data.
    
    Args:
        processed_data: Numpy array of shape [num_stocks, num_days, num_features]
        tickers: List of tickers corresponding to the stocks
        output_path: Path to save the visualization
        num_samples: Number of stocks to visualize
    """
    # Select random stocks to visualize
    num_stocks = len(tickers)
    sample_indices = np.random.choice(
        range(num_stocks), 
        size=min(num_samples, num_stocks), 
        replace=False
    )
    
    # Feature names for better visualization
    feature_names = ["Close", "High", "Low", "Volume"]
    num_features = min(len(feature_names), processed_data.shape[2])
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=len(sample_indices), 
        ncols=num_features, 
        figsize=(15, 3 * len(sample_indices))
    )
    
    # Ensure axes is 2D even if num_samples=1 or num_features=1
    if len(sample_indices) == 1:
        axes = [axes]
    if num_features == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each feature for each stock
    for i, stock_idx in enumerate(sample_indices):
        for j in range(num_features):
            ax = axes[i][j]
            feature_data = processed_data[stock_idx, :, j]
            
            ax.plot(feature_data)
            ax.set_title(f"{tickers[stock_idx]} - {feature_names[j]}")
            
            if i == len(sample_indices) - 1:
                ax.set_xlabel("Days")
            
            if j == 0:
                ax.set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")


def create_industry_matrix(tickers: List[str]) -> np.ndarray:
    """
    Create an industry classification matrix based on GICS sectors.
    
    Args:
        tickers: List of stock tickers
        
    Returns:
        Industry matrix of shape [num_stocks, num_stocks]
    """
    # Define industry groups (simplified GICS sectors)
    industry_groups = {
        'Technology': ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'NVDA', 'AMD', 'INTC', 'CSCO', 'CRM', 'ADBE', 'PYPL', 'QCOM', 'TXN'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'AMGN', 'GILD', 'BIIB', 'ISRG', 'MDT', 'TMO', 'UNH'],
        'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA', 'C', 'USB', 'MET'],
        'Consumer': ['AMZN', 'WMT', 'COST', 'MCD', 'SBUX', 'NKE', 'TGT', 'HD', 'LOW', 'DIS', 'NFLX'],
        'Industrial': ['BA', 'CAT', 'GE', 'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'GD', 'EMR'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'VLO', 'MPC', 'PSX'],
        'Materials': ['LIN', 'ECL', 'NEM', 'FCX', 'NUE', 'BLL', 'ALB', 'SHW'],
        'Utilities': ['DUK', 'SO', 'NEE', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
        'Real Estate': ['AMT', 'CCI', 'PLD', 'WELL', 'SPG', 'PSA', 'EQR', 'AVB'],
        'Communication': ['T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'DISH', 'FOX', 'FOXA']
    }
    
    # Create industry matrix
    num_stocks = len(tickers)
    industry_matrix = np.zeros((num_stocks, num_stocks))
    
    # Map tickers to their industries
    ticker_to_industry = {}
    for industry, industry_tickers in industry_groups.items():
        for ticker in industry_tickers:
            if ticker in tickers:
                ticker_to_industry[ticker] = industry
    
    # Fill the matrix
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if ticker1 in ticker_to_industry and ticker2 in ticker_to_industry:
                if ticker_to_industry[ticker1] == ticker_to_industry[ticker2]:
                    industry_matrix[i, j] = 1
    
    # Normalize the matrix
    row_sums = industry_matrix.sum(axis=1, keepdims=True)
    industry_matrix = np.divide(industry_matrix, row_sums, where=row_sums!=0)
    
    return industry_matrix


def download_and_process_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    min_history: int = 252,
    required_days: int = None,
    create_relation: bool = True,
    visualize: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Download, process, and save stock data in one function.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save the processed data
        min_history: Minimum history length required
        required_days: Optional specific number of days required
        create_relation: Whether to create and save a correlation matrix
        visualize: Whether to create and save visualization
        
    Returns:
        Tuple of (processed_data, valid_tickers)
    """
    # Download data
    stock_data = download_stock_data(tickers, start_date, end_date)
    
    # Process data
    processed_data, valid_tickers = preprocess_stock_data(
        stock_data, 
        min_history=min_history,
        required_days=required_days
    )
    
    # Create industry matrix
    industry_matrix = create_industry_matrix(valid_tickers)
    
    # Save data
    save_processed_data(processed_data, valid_tickers, output_dir, industry_matrix)
    
    # Create visualization if requested
    if visualize:
        vis_path = os.path.join(output_dir, "stock_data_visualization.png")
        visualize_stock_data(processed_data, valid_tickers, vis_path)
    
    return processed_data, valid_tickers


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys
    import os
    
    # Add project root to path to allow importing from configs
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description="Download and process stock data for RL trading")
    parser.add_argument("--tickers", type=str, help="Path to file containing ticker symbols, one per line")
    parser.add_argument("--start-date", type=str, default="2000-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=None, help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--min-history", type=int, default=252, help="Minimum history length required (days)")
    parser.add_argument("--required-days", type=int, default=None, help="Specific number of days required")
    parser.add_argument("--no-relation", action="store_true", help="Do not create correlation matrix")
    parser.add_argument("--no-visualize", action="store_true", help="Do not create visualization")
    
    args = parser.parse_args()
    
    # Set default end date to today if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Import tickers from the Python file
    from configs.target_tickers import TARGET_TICKERS
    tickers = TARGET_TICKERS
    
    # Download and process data
    download_and_process_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        min_history=args.min_history,
        required_days=args.required_days,
        create_relation=not args.no_relation,
        visualize=not args.no_visualize
    )


