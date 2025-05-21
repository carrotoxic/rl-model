import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Utility class for preprocessing financial data with missing values.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the data preprocessor.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file handler for logging
        self.setup_logging()
        
        # Tracking metrics
        self.missing_data_stats = {}
        
    def setup_logging(self):
        """Configure logging for the data preprocessor."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'data_preprocessing.log')
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def detect_missing_data(
        self, 
        data: np.ndarray, 
        ticker_list: List[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect missing data (NaN or inf) in the input data.
        
        Args:
            data: Input data array [num_stocks, num_days, num_features] or [num_stocks, num_days]
            ticker_list: List of ticker symbols (optional)
            
        Returns:
            Tuple of (mask of missing values, statistics about missing data)
        """
        # Create missing value mask (True where data is missing)
        missing_mask = np.isnan(data) | np.isinf(data)
        
        # Calculate statistics
        total_elements = data.size
        missing_elements = np.sum(missing_mask)
        missing_percentage = (missing_elements / total_elements) * 100
        
        # Create statistics dictionary
        stats = {
            'total_elements': total_elements,
            'missing_elements': missing_elements,
            'missing_percentage': missing_percentage,
        }
        
        if data.ndim == 3:
            # For 3D data [stocks, days, features]
            num_stocks, num_days, num_features = data.shape
            
            # Calculate missing values per stock
            missing_per_stock = np.sum(missing_mask, axis=(1, 2))
            missing_percent_per_stock = (missing_per_stock / (num_days * num_features)) * 100
            
            # Calculate missing values per day
            missing_per_day = np.sum(missing_mask, axis=(0, 2))
            missing_percent_per_day = (missing_per_day / (num_stocks * num_features)) * 100
            
            # Calculate missing values per feature
            missing_per_feature = np.sum(missing_mask, axis=(0, 1))
            missing_percent_per_feature = (missing_per_feature / (num_stocks * num_days)) * 100
            
            stats.update({
                'missing_per_stock': missing_per_stock,
                'missing_percent_per_stock': missing_percent_per_stock,
                'missing_per_day': missing_per_day,
                'missing_percent_per_day': missing_percent_per_day,
                'missing_per_feature': missing_per_feature,
                'missing_percent_per_feature': missing_percent_per_feature,
            })
            
        elif data.ndim == 2:
            # For 2D data [stocks, days]
            num_stocks, num_days = data.shape
            
            # Calculate missing values per stock
            missing_per_stock = np.sum(missing_mask, axis=1)
            missing_percent_per_stock = (missing_per_stock / num_days) * 100
            
            # Calculate missing values per day
            missing_per_day = np.sum(missing_mask, axis=0)
            missing_percent_per_day = (missing_per_day / num_stocks) * 100
            
            stats.update({
                'missing_per_stock': missing_per_stock,
                'missing_percent_per_stock': missing_percent_per_stock,
                'missing_per_day': missing_per_day,
                'missing_percent_per_day': missing_percent_per_day,
            })
            
        # Add ticker information if provided
        if ticker_list is not None:
            missing_tickers = {}
            for i, ticker in enumerate(ticker_list):
                if data.ndim == 3:
                    ticker_missing = np.sum(missing_mask[i, :, :])
                    if ticker_missing > 0:
                        missing_tickers[ticker] = ticker_missing
                elif data.ndim == 2:
                    ticker_missing = np.sum(missing_mask[i, :])
                    if ticker_missing > 0:
                        missing_tickers[ticker] = ticker_missing
            
            stats['missing_tickers'] = missing_tickers
        
        # Log statistics
        logger.info(f"Missing data analysis completed. {missing_percentage:.2f}% of data is missing.")
        if ticker_list is not None and missing_tickers:
            logger.info(f"Tickers with missing data: {', '.join(missing_tickers.keys())}")
        
        return missing_mask, stats
    
    def impute_missing_data(
        self, 
        data: np.ndarray, 
        method: str = 'forward_fill',
        fill_value: float = 0.0,
        ticker_list: List[str] = None,
        missing_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Impute missing data in the input array.
        
        Args:
            data: Input data array
            method: Imputation method ('zero', 'mean', 'median', 'forward_fill', 'backward_fill')
            fill_value: Value to use for 'constant' method
            ticker_list: List of ticker symbols (optional)
            missing_mask: Pre-computed mask of missing values (optional)
            
        Returns:
            Tuple of (imputed data, imputation mask, imputation statistics)
        """
        # Convert to float array to ensure NaN handling
        data = data.astype(np.float32)
        
        # Detect missing data if mask not provided
        if missing_mask is None:
            missing_mask, _ = self.detect_missing_data(data, ticker_list)
        
        # Create a copy of the data to avoid modifying the original
        imputed_data = data.copy()
        
        # Create mask to track which values are imputed
        imputation_mask = np.zeros_like(missing_mask, dtype=np.int8)
        
        # Track statistics
        stats = {
            'method': method,
            'total_imputed': np.sum(missing_mask),
        }
        
        # Perform imputation based on method
        if method == 'zero' or method == 'constant':
            imputed_data[missing_mask] = fill_value
            imputation_mask[missing_mask] = 1
            
            logger.info(f"Imputed {stats['total_imputed']} missing values with {fill_value} ({method} method)")
            
        elif method == 'mean':
            if data.ndim == 3:
                # Handle 3D data (stocks, days, features)
                for stock_idx in range(data.shape[0]):
                    for feature_idx in range(data.shape[2]):
                        feature_data = data[stock_idx, :, feature_idx]
                        feature_mask = missing_mask[stock_idx, :, feature_idx]
                        
                        if np.all(feature_mask):
                            # All values missing, use global mean
                            global_mean = np.nanmean(data[:, :, feature_idx])
                            imputed_data[stock_idx, feature_mask, feature_idx] = global_mean
                        else:
                            # Some values available, use stock-feature mean
                            mean_value = np.nanmean(feature_data)
                            imputed_data[stock_idx, feature_mask, feature_idx] = mean_value
                        
                        imputation_mask[stock_idx, feature_mask, feature_idx] = 1
            
            elif data.ndim == 2:
                # Handle 2D data (stocks, days)
                for stock_idx in range(data.shape[0]):
                    stock_data = data[stock_idx, :]
                    stock_mask = missing_mask[stock_idx, :]
                    
                    if np.all(stock_mask):
                        # All values missing, use global mean
                        global_mean = np.nanmean(data)
                        imputed_data[stock_idx, stock_mask] = global_mean
                    else:
                        # Some values available, use stock mean
                        mean_value = np.nanmean(stock_data)
                        imputed_data[stock_idx, stock_mask] = mean_value
                    
                    imputation_mask[stock_idx, stock_mask] = 1
                    
            logger.info(f"Imputed {stats['total_imputed']} missing values with mean values")
            
        elif method == 'median':
            if data.ndim == 3:
                # Handle 3D data
                for stock_idx in range(data.shape[0]):
                    for feature_idx in range(data.shape[2]):
                        feature_data = data[stock_idx, :, feature_idx]
                        feature_mask = missing_mask[stock_idx, :, feature_idx]
                        
                        if np.all(feature_mask):
                            # All values missing, use global median
                            global_median = np.nanmedian(data[:, :, feature_idx])
                            imputed_data[stock_idx, feature_mask, feature_idx] = global_median
                        else:
                            # Some values available, use stock-feature median
                            median_value = np.nanmedian(feature_data)
                            imputed_data[stock_idx, feature_mask, feature_idx] = median_value
                        
                        imputation_mask[stock_idx, feature_mask, feature_idx] = 1
            
            elif data.ndim == 2:
                # Handle 2D data
                for stock_idx in range(data.shape[0]):
                    stock_data = data[stock_idx, :]
                    stock_mask = missing_mask[stock_idx, :]
                    
                    if np.all(stock_mask):
                        # All values missing, use global median
                        global_median = np.nanmedian(data)
                        imputed_data[stock_idx, stock_mask] = global_median
                    else:
                        # Some values available, use stock median
                        median_value = np.nanmedian(stock_data)
                        imputed_data[stock_idx, stock_mask] = median_value
                    
                    imputation_mask[stock_idx, stock_mask] = 1
                    
            logger.info(f"Imputed {stats['total_imputed']} missing values with median values")
            
        elif method == 'forward_fill':
            # Convert to pandas for easier forward fill
            if data.ndim == 3:
                # For 3D data, handle each stock and feature separately
                for stock_idx in range(data.shape[0]):
                    for feature_idx in range(data.shape[2]):
                        # Extract series and mask
                        series = data[stock_idx, :, feature_idx]
                        series_mask = missing_mask[stock_idx, :, feature_idx]
                        
                        if np.any(series_mask):
                            # Convert to pandas Series for ffill
                            pd_series = pd.Series(series)
                            filled_series = pd_series.ffill()
                            
                            # Handle case where first values are NaN (can't forward fill)
                            if filled_series.isna().any():
                                filled_series = filled_series.bfill()
                                
                            # Update the imputed data
                            imputed_data[stock_idx, :, feature_idx] = filled_series.values
                            
                            # Update imputation mask
                            imputation_mask[stock_idx, series_mask, feature_idx] = 1
            
            elif data.ndim == 2:
                # For 2D data, handle each stock separately
                for stock_idx in range(data.shape[0]):
                    # Extract series and mask
                    series = data[stock_idx, :]
                    series_mask = missing_mask[stock_idx, :]
                    
                    if np.any(series_mask):
                        # Convert to pandas Series for ffill
                        pd_series = pd.Series(series)
                        filled_series = pd_series.ffill()
                        
                        # Handle case where first values are NaN (can't forward fill)
                        if filled_series.isna().any():
                            filled_series = filled_series.bfill()
                            
                        # Update the imputed data
                        imputed_data[stock_idx, :] = filled_series.values
                        
                        # Update imputation mask
                        imputation_mask[stock_idx, series_mask] = 1
                        
            logger.info(f"Imputed {stats['total_imputed']} missing values with forward fill")
            
        elif method == 'backward_fill':
            # Similar to forward fill but using backward fill
            if data.ndim == 3:
                for stock_idx in range(data.shape[0]):
                    for feature_idx in range(data.shape[2]):
                        series = data[stock_idx, :, feature_idx]
                        series_mask = missing_mask[stock_idx, :, feature_idx]
                        
                        if np.any(series_mask):
                            pd_series = pd.Series(series)
                            filled_series = pd_series.bfill()
                            
                            # Handle case where last values are NaN
                            if filled_series.isna().any():
                                filled_series = filled_series.ffill()
                                
                            imputed_data[stock_idx, :, feature_idx] = filled_series.values
                            imputation_mask[stock_idx, series_mask, feature_idx] = 1
            
            elif data.ndim == 2:
                for stock_idx in range(data.shape[0]):
                    series = data[stock_idx, :]
                    series_mask = missing_mask[stock_idx, :]
                    
                    if np.any(series_mask):
                        pd_series = pd.Series(series)
                        filled_series = pd_series.bfill()
                        
                        # Handle case where last values are NaN
                        if filled_series.isna().any():
                            filled_series = filled_series.ffill()
                            
                        imputed_data[stock_idx, :] = filled_series.values
                        imputation_mask[stock_idx, series_mask] = 1
                        
            logger.info(f"Imputed {stats['total_imputed']} missing values with backward fill")
            
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # Verify no NaN values remain
        if np.isnan(imputed_data).any() or np.isinf(imputed_data).any():
            remaining_missing = np.sum(np.isnan(imputed_data) | np.isinf(imputed_data))
            logger.warning(f"{remaining_missing} missing values remain after imputation!")
            
            # As a fallback, replace any remaining NaN with zeros
            remaining_mask = np.isnan(imputed_data) | np.isinf(imputed_data)
            imputed_data[remaining_mask] = 0.0
            imputation_mask[remaining_mask] = 2  # Mark as fallback imputation
            
            logger.info(f"Applied fallback imputation (zero) to remaining missing values")
        
        # Store statistics
        stats['imputation_mask'] = imputation_mask
        
        # Log ticker-specific information if provided
        if ticker_list is not None:
            imputed_tickers = {}
            for i, ticker in enumerate(ticker_list):
                if data.ndim == 3:
                    ticker_imputed = np.sum(imputation_mask[i, :, :])
                    if ticker_imputed > 0:
                        imputed_tickers[ticker] = ticker_imputed
                elif data.ndim == 2:
                    ticker_imputed = np.sum(imputation_mask[i, :])
                    if ticker_imputed > 0:
                        imputed_tickers[ticker] = ticker_imputed
            
            stats['imputed_tickers'] = imputed_tickers
            
            if imputed_tickers:
                logger.info(f"Imputed data for tickers: {', '.join(imputed_tickers.keys())}")
                
                # Detailed logging for tickers with high missing data
                high_missing_tickers = []
                for ticker, count in imputed_tickers.items():
                    if data.ndim == 3:
                        ticker_idx = ticker_list.index(ticker)
                        total_elements = data.shape[1] * data.shape[2]
                        missing_percent = (count / total_elements) * 100
                        if missing_percent > 30:
                            high_missing_tickers.append((ticker, missing_percent))
                    elif data.ndim == 2:
                        ticker_idx = ticker_list.index(ticker)
                        total_elements = data.shape[1]
                        missing_percent = (count / total_elements) * 100
                        if missing_percent > 30:
                            high_missing_tickers.append((ticker, missing_percent))
                
                if high_missing_tickers:
                    for ticker, percent in high_missing_tickers:
                        logger.warning(f"Ticker {ticker} has {percent:.1f}% missing data")
        
        return imputed_data, imputation_mask, stats
    
    def handle_missing_tickers(
        self,
        data: np.ndarray,
        ticker_list: List[str],
        min_data_threshold: float = 0.7,
        impute_method: str = 'forward_fill'
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Handle missing tickers by either imputing their data or removing them.
        
        Args:
            data: Input data array [num_stocks, num_days, num_features] or [num_stocks, num_days]
            ticker_list: List of ticker symbols
            min_data_threshold: Minimum fraction of data points required to keep a ticker
            impute_method: Method to impute missing values if ticker is kept
            
        Returns:
            Tuple of (processed data, list of kept tickers, list of removed tickers)
        """
        if len(ticker_list) != data.shape[0]:
            raise ValueError(f"Ticker list length ({len(ticker_list)}) doesn't match data shape ({data.shape[0]})")
        
        # Detect missing data
        missing_mask, stats = self.detect_missing_data(data, ticker_list)
        
        # Calculate missing percentage per ticker
        if data.ndim == 3:
            missing_per_ticker = np.sum(missing_mask, axis=(1, 2))
            total_elements_per_ticker = data.shape[1] * data.shape[2]
        else:  # 2D
            missing_per_ticker = np.sum(missing_mask, axis=1)
            total_elements_per_ticker = data.shape[1]
        
        missing_percent_per_ticker = missing_per_ticker / total_elements_per_ticker
        
        # Determine which tickers to keep
        keep_mask = missing_percent_per_ticker <= (1 - min_data_threshold)
        tickers_to_keep = [ticker for i, ticker in enumerate(ticker_list) if keep_mask[i]]
        tickers_to_remove = [ticker for i, ticker in enumerate(ticker_list) if not keep_mask[i]]
        
        # Log the decision
        logger.info(f"Keeping {len(tickers_to_keep)}/{len(ticker_list)} tickers")
        if tickers_to_remove:
            logger.warning(f"Removing {len(tickers_to_remove)} tickers due to insufficient data: {', '.join(tickers_to_remove)}")
            
            # Log detailed statistics for removed tickers
            for ticker in tickers_to_remove:
                idx = ticker_list.index(ticker)
                percent_missing = missing_percent_per_ticker[idx] * 100
                logger.info(f"Ticker {ticker} removed: {percent_missing:.1f}% missing data")
        
        # Extract data for kept tickers
        kept_data = data[keep_mask]
        
        # Impute missing values for kept tickers
        if np.any(missing_mask[keep_mask]):
            kept_tickers = [ticker for i, ticker in enumerate(ticker_list) if keep_mask[i]]
            logger.info(f"Imputing missing values for kept tickers using {impute_method} method")
            
            # Only pass the part of the missing mask that corresponds to kept tickers
            kept_missing_mask = missing_mask[keep_mask]
            
            imputed_data, _, impute_stats = self.impute_missing_data(
                kept_data, 
                method=impute_method,
                ticker_list=kept_tickers,
                missing_mask=kept_missing_mask
            )
            
            # Use imputed data
            processed_data = imputed_data
        else:
            # No missing values in kept tickers
            processed_data = kept_data
            logger.info("No missing values in kept tickers, no imputation needed")
        
        return processed_data, tickers_to_keep, tickers_to_remove
    
    def save_missing_data_report(
        self,
        stats: Dict,
        ticker_list: List[str] = None,
        output_file: str = None
    ) -> None:
        """
        Save a detailed report about missing data.
        
        Args:
            stats: Statistics dictionary from detect_missing_data
            ticker_list: List of ticker symbols
            output_file: File path to save the report
        """
        if output_file is None:
            output_file = os.path.join(self.log_dir, 'missing_data_report.txt')
        
        with open(output_file, 'w') as f:
            f.write("==== Missing Data Report ====\n\n")
            
            f.write(f"Total elements: {stats['total_elements']}\n")
            f.write(f"Missing elements: {stats['missing_elements']}\n")
            f.write(f"Missing percentage: {stats['missing_percentage']:.2f}%\n\n")
            
            if 'missing_tickers' in stats and ticker_list is not None:
                f.write("==== Tickers with Missing Data ====\n\n")
                
                # Sort tickers by missing count (descending)
                sorted_tickers = sorted(
                    stats['missing_tickers'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                f.write(f"{'Ticker':<10} {'Missing Count':<15} {'Missing Percentage':<20}\n")
                f.write(f"{'-'*10} {'-'*15} {'-'*20}\n")
                
                for ticker, count in sorted_tickers:
                    idx = ticker_list.index(ticker)
                    if 'missing_percent_per_stock' in stats:
                        percent = stats['missing_percent_per_stock'][idx]
                        f.write(f"{ticker:<10} {count:<15} {percent:.2f}%\n")
                    else:
                        f.write(f"{ticker:<10} {count:<15}\n")
                
                f.write("\n")
            
            if 'missing_per_day' in stats:
                f.write("==== Missing Data by Day ====\n\n")
                
                days_with_missing = np.where(stats['missing_per_day'] > 0)[0]
                if len(days_with_missing) > 0:
                    f.write(f"Days with missing data: {len(days_with_missing)}\n")
                    
                    # Identify periods with consecutive missing days
                    f.write("Periods with significant missing data:\n")
                    
                    # Simple algorithm to find ranges of days with high missing rates
                    threshold = 0.1  # 10% or more stocks missing
                    high_missing_days = np.where(stats['missing_percent_per_day'] > threshold * 100)[0]
                    
                    if len(high_missing_days) > 0:
                        ranges = []
                        start = high_missing_days[0]
                        
                        for i in range(1, len(high_missing_days)):
                            if high_missing_days[i] > high_missing_days[i-1] + 1:
                                # Gap found, end current range
                                ranges.append((start, high_missing_days[i-1]))
                                start = high_missing_days[i]
                        
                        # Add the last range
                        ranges.append((start, high_missing_days[-1]))
                        
                        for start_day, end_day in ranges:
                            duration = end_day - start_day + 1
                            avg_missing = np.mean(stats['missing_percent_per_day'][start_day:end_day+1])
                            f.write(f"  Days {start_day}-{end_day} ({duration} days): {avg_missing:.2f}% missing on average\n")
                    else:
                        f.write("  No periods with significant missing data\n")
                    
                    f.write("\n")
                else:
                    f.write("No days with missing data\n\n")
            
            if 'missing_per_feature' in stats:
                f.write("==== Missing Data by Feature ====\n\n")
                
                features_with_missing = np.where(stats['missing_per_feature'] > 0)[0]
                if len(features_with_missing) > 0:
                    f.write(f"Features with missing data: {len(features_with_missing)}\n")
                    f.write(f"{'Feature ID':<10} {'Missing Count':<15} {'Missing Percentage':<20}\n")
                    f.write(f"{'-'*10} {'-'*15} {'-'*20}\n")
                    
                    for feature_id in features_with_missing:
                        count = stats['missing_per_feature'][feature_id]
                        percent = stats['missing_percent_per_feature'][feature_id]
                        f.write(f"{feature_id:<10} {count:<15} {percent:.2f}%\n")
                    
                    f.write("\n")
                else:
                    f.write("No features with missing data\n\n")
        
        logger.info(f"Missing data report saved to {output_file}")


def validate_data_integrity(
    data: np.ndarray, 
    name: str = "data",
    log_issues: bool = True
) -> bool:
    """
    Validate the integrity of a data array.
    
    Args:
        data: The data array to validate
        name: Name of the data array for logging
        log_issues: Whether to log issues found
        
    Returns:
        True if data passes all checks, False otherwise
    """
    valid = True
    
    # Check for NaN and infinity
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    
    if nan_count > 0:
        valid = False
        if log_issues:
            logger.error(f"{name} contains {nan_count} NaN values")
    
    if inf_count > 0:
        valid = False
        if log_issues:
            logger.error(f"{name} contains {inf_count} infinite values")
    
    # Check for unreasonable values (depends on the specific data)
    # Example: extremely large magnitude values
    large_vals = np.sum(np.abs(data) > 1e6)
    if large_vals > 0:
        # Just a warning, not invalidating the data
        if log_issues:
            logger.warning(f"{name} contains {large_vals} values with magnitude > 1e6")
    
    # Check for all zeros in a dimension (could indicate improperly processed data)
    if data.ndim == 3:
        # For 3D data (stocks, days, features)
        for i in range(data.shape[0]):
            if np.all(data[i] == 0):
                if log_issues:
                    logger.warning(f"{name}: Stock {i} has all zero values")
        
        for i in range(data.shape[2]):
            if np.all(data[:, :, i] == 0):
                if log_issues:
                    logger.warning(f"{name}: Feature {i} has all zero values")
    
    elif data.ndim == 2:
        # For 2D data (stocks, days)
        for i in range(data.shape[0]):
            if np.all(data[i] == 0):
                if log_issues:
                    logger.warning(f"{name}: Stock {i} has all zero values")
    
    return valid 