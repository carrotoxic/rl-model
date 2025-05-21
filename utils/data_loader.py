import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional, List, Union
from utils.data_preprocessing import DataPreprocessor, validate_data_integrity


# Set up logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for loading and preprocessing market and stock data.
    """
    
    def __init__(self, data_dir: str, market_name: str, log_dir: str = "logs"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data
            market_name: Name of the market (e.g., "DJIA")
            log_dir: Directory to save logs
        """
        self.data_dir = data_dir
        self.market_name = market_name
        self.market_dir = os.path.join(data_dir, market_name)
        self.log_dir = log_dir
        
        # Set up logging
        self.setup_logging()
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor(log_dir=log_dir)
        
        # Check if market directory exists
        if not os.path.exists(self.market_dir):
            logger.error(f"Market directory {self.market_dir} not found")
            raise FileNotFoundError(f"Market directory {self.market_dir} not found")
        else:
            logger.info(f"Found market directory: {self.market_dir}")
            
    def setup_logging(self):
        """Configure logging for the data loader."""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'data_loader.log')
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def load_ticker_metadata(self, ticker_file: Optional[str] = "tickers.txt") -> List[str]:
        """
        Load ticker metadata from a file.
        
        Args:
            ticker_file: File containing ticker symbols
            
        Returns:
            List of ticker symbols
        """
        ticker_path = os.path.join(self.market_dir, ticker_file)
        
        if os.path.exists(ticker_path):
            with open(ticker_path, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(tickers)} tickers from {ticker_path}")
            return tickers
        else:
            # If no ticker file is found, we'll try to infer from data
            logger.warning(f"Ticker file {ticker_path} not found, will infer tickers from data")
            return []
    
    def load_data(
        self,
        window_size: int = 13,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        normalize: bool = True,
        relation_file: Optional[str] = "relation_matrix.npy",
        min_data_threshold: float = 0.7,
        impute_method: str = "forward_fill",
        ticker_file: Optional[str] = "tickers.txt"
    ) -> Dict:
        """
        Load and preprocess data.
        
        Args:
            window_size: Size of the input window
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            normalize: Whether to normalize data
            relation_file: Name of relation matrix file
            min_data_threshold: Minimum fraction of data required to keep a ticker
            impute_method: Method to impute missing values
            ticker_file: File containing ticker symbols
            
        Returns:
            Dictionary containing processed data
        """
        logger.info(f"Loading data for market {self.market_name}")
        
        # Load ticker metadata
        tickers = self.load_ticker_metadata(ticker_file)
        
        # Load stock data
        stock_data_path = os.path.join(self.market_dir, "stocks_data.npy")
        if not os.path.exists(stock_data_path):
            logger.error(f"Stock data file {stock_data_path} not found")
            raise FileNotFoundError(f"Stock data file {stock_data_path} not found")
        
        logger.info(f"Loading stock data from {stock_data_path}")
        stock_data = np.load(stock_data_path)  # [num_stocks, num_days, features]
        
        # Load market data
        market_data_path = os.path.join(self.market_dir, "market_data.npy")
        if not os.path.exists(market_data_path):
            logger.error(f"Market data file {market_data_path} not found")
            raise FileNotFoundError(f"Market data file {market_data_path} not found")
        
        logger.info(f"Loading market data from {market_data_path}")
        market_data = np.load(market_data_path)  # [num_days, features]
        
        # Load returns data
        returns_data_path = os.path.join(self.market_dir, "ror.npy")
        if not os.path.exists(returns_data_path):
            logger.error(f"Returns data file {returns_data_path} not found")
            raise FileNotFoundError(f"Returns data file {returns_data_path} not found")
        
        logger.info(f"Loading returns data from {returns_data_path}")
        returns_data = np.load(returns_data_path)  # [num_stocks, num_days]
        
        # Load relation matrix if provided
        relation_matrix = None
        if relation_file is not None:
            relation_matrix_path = os.path.join(self.market_dir, relation_file)
            if os.path.exists(relation_matrix_path):
                logger.info(f"Loading relation matrix from {relation_matrix_path}")
                relation_matrix = np.load(relation_matrix_path)
            else:
                logger.warning(f"Relation matrix file {relation_matrix_path} not found")
        
        # Get dimensions
        num_stocks, num_days, stock_features = stock_data.shape
        market_features = market_data.shape[1]
        
        # Validate shape consistency
        if returns_data.shape[0] != num_stocks or returns_data.shape[1] != num_days:
            logger.error(f"Shape mismatch: stock_data {stock_data.shape}, returns_data {returns_data.shape}")
            raise ValueError(f"Shape mismatch: stock_data {stock_data.shape}, returns_data {returns_data.shape}")
        
        if market_data.shape[0] != num_days:
            logger.error(f"Shape mismatch: market_data {market_data.shape}, days in stock_data {num_days}")
            raise ValueError(f"Shape mismatch: market_data {market_data.shape}, days in stock_data {num_days}")
        
        # If no tickers provided, generate placeholder names
        if not tickers:
            tickers = [f"TICKER_{i}" for i in range(num_stocks)]
            logger.info(f"Generated {len(tickers)} placeholder ticker names")
        
        # Validate data integrity
        if not validate_data_integrity(stock_data, "stock_data"):
            logger.warning("Stock data contains invalid values, will be processed")
        
        if not validate_data_integrity(market_data, "market_data"):
            logger.warning("Market data contains invalid values, will be processed")
            
        if not validate_data_integrity(returns_data, "returns_data"):
            logger.warning("Returns data contains invalid values, will be processed")
        
        # Handle missing data in stock data
        logger.info("Processing stock data for missing values")
        processed_stock_data, kept_tickers, removed_tickers = self.preprocessor.handle_missing_tickers(
            stock_data,
            tickers,
            min_data_threshold=min_data_threshold,
            impute_method=impute_method
        )
        
        # Update returns data to match kept tickers
        keep_indices = [tickers.index(ticker) for ticker in kept_tickers]
        processed_returns_data = returns_data[keep_indices]
        
        # Handle missing values in processed returns data
        if not validate_data_integrity(processed_returns_data, "processed_returns_data"):
            logger.info("Processing returns data for missing values")
            
            # Detect missing values
            missing_mask, stats = self.preprocessor.detect_missing_data(
                processed_returns_data, 
                kept_tickers
            )
            
            # Save missing data report
            self.preprocessor.save_missing_data_report(
                stats,
                kept_tickers,
                os.path.join(self.log_dir, 'returns_missing_data_report.txt')
            )
            
            # Impute missing values
            processed_returns_data, _, _ = self.preprocessor.impute_missing_data(
                processed_returns_data,
                method=impute_method,
                ticker_list=kept_tickers,
                missing_mask=missing_mask
            )
        
        # Handle missing values in market data
        if not validate_data_integrity(market_data, "market_data"):
            logger.info("Processing market data for missing values")
            
            # Detect missing values
            missing_mask, stats = self.preprocessor.detect_missing_data(market_data)
            
            # Save missing data report
            self.preprocessor.save_missing_data_report(
                stats,
                output_file=os.path.join(self.log_dir, 'market_missing_data_report.txt')
            )
            
            # Impute missing values
            market_data, _, _ = self.preprocessor.impute_missing_data(
                market_data,
                method=impute_method,
                missing_mask=missing_mask
            )
        
        # Update relation matrix if needed
        if relation_matrix is not None:
            if len(kept_tickers) != num_stocks:
                logger.info("Updating relation matrix to match kept tickers")
                
                # Extract submatrix for kept tickers
                keep_indices = np.array([tickers.index(ticker) for ticker in kept_tickers])
                relation_matrix = relation_matrix[np.ix_(keep_indices, keep_indices)]
                
                # Validate the updated relation matrix
                if relation_matrix.shape != (len(kept_tickers), len(kept_tickers)):
                    logger.error(f"Relation matrix shape mismatch: {relation_matrix.shape}, expected {(len(kept_tickers), len(kept_tickers))}")
                    raise ValueError(f"Relation matrix shape mismatch: {relation_matrix.shape}, expected {(len(kept_tickers), len(kept_tickers))}")
        
        # Update num_stocks after filtering
        num_stocks = len(kept_tickers)
        
        # Normalize data if requested
        if normalize:
            logger.info("Normalizing data")
            
            # Normalize stock data
            for s in range(num_stocks):
                for f in range(stock_features):
                    feature_data = processed_stock_data[s, :, f]
                    mean, std = np.mean(feature_data), np.std(feature_data)
                    if std > 0:
                        processed_stock_data[s, :, f] = (feature_data - mean) / std
            
            # Normalize market data
            for f in range(market_features):
                feature_data = market_data[:, f]
                mean, std = np.mean(feature_data), np.std(feature_data)
                if std > 0:
                    market_data[:, f] = (feature_data - mean) / std
                    
            logger.info("Data normalization completed")
        
        # Split data into train, validation, and test sets
        logger.info(f"Splitting data with ratios - train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
        train_end = int(num_days * train_ratio)
        val_end = int(num_days * (train_ratio + val_ratio))
        
        train_indices = range(window_size, train_end)
        val_indices = range(train_end, val_end)
        test_indices = range(val_end, num_days)
        
        logger.info(f"Split sizes - train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")
        
        # Prepare data dictionary
        data_dict = {
            'stock_data': processed_stock_data,
            'market_data': market_data,
            'returns_data': processed_returns_data,
            'relation_matrix': relation_matrix,
            'tickers': kept_tickers,
            'removed_tickers': removed_tickers,
            'num_stocks': num_stocks,
            'num_days': num_days,
            'stock_features': stock_features,
            'market_features': market_features,
            'window_size': window_size,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'dataset_metadata': {
                'original_num_stocks': len(tickers),
                'removed_stocks': len(removed_tickers),
                'min_data_threshold': min_data_threshold,
                'impute_method': impute_method
            }
        }
        
        logger.info(f"Data loading completed successfully. Loaded {num_stocks} stocks with {num_days} days of data.")
        
        return data_dict
    
    def create_observation(
        self,
        data_dict: Dict,
        idx: int
    ) -> Dict:
        """
        Create observation at a specific index.
        
        Args:
            data_dict: Data dictionary from load_data
            idx: Current index
            
        Returns:
            Observation dictionary
        """
        window_size = data_dict['window_size']
        
        # Extract window of data
        stock_window = data_dict['stock_data'][:, idx - window_size:idx, :]
        market_window = data_dict['market_data'][idx - window_size:idx, :]
        
        # Create observation dictionary
        observation = {
            'stocks': stock_window.transpose(1, 0, 2),  # [window, stocks, features]
            'market': market_window,
            'portfolio': np.zeros(data_dict['num_stocks'])  # Initial empty portfolio
        }
        
        return observation
    
    def save_processed_data(self, data_dict: Dict, output_dir: Optional[str] = None) -> None:
        """
        Save processed data to files.
        
        Args:
            data_dict: Processed data dictionary
            output_dir: Directory to save processed data
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, f"{self.market_name}_processed")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save stock data
        np.save(os.path.join(output_dir, "stocks_data.npy"), data_dict['stock_data'])
        
        # Save market data
        np.save(os.path.join(output_dir, "market_data.npy"), data_dict['market_data'])
        
        # Save returns data
        np.save(os.path.join(output_dir, "ror.npy"), data_dict['returns_data'])
        
        # Save relation matrix if available
        if data_dict['relation_matrix'] is not None:
            np.save(os.path.join(output_dir, "relation_matrix.npy"), data_dict['relation_matrix'])
        
        # Save ticker information
        with open(os.path.join(output_dir, "tickers.txt"), 'w') as f:
            for ticker in data_dict['tickers']:
                f.write(f"{ticker}\n")
        
        # Save removed tickers if any
        if data_dict['removed_tickers']:
            with open(os.path.join(output_dir, "removed_tickers.txt"), 'w') as f:
                for ticker in data_dict['removed_tickers']:
                    f.write(f"{ticker}\n")
        
        # Save metadata
        with open(os.path.join(output_dir, "metadata.txt"), 'w') as f:
            f.write(f"Market: {self.market_name}\n")
            f.write(f"Number of stocks: {data_dict['num_stocks']}\n")
            f.write(f"Number of days: {data_dict['num_days']}\n")
            f.write(f"Stock features: {data_dict['stock_features']}\n")
            f.write(f"Market features: {data_dict['market_features']}\n")
            f.write(f"Window size: {data_dict['window_size']}\n")
            f.write(f"Train indices: {data_dict['train_indices'][0]}-{data_dict['train_indices'][-1]}\n")
            f.write(f"Validation indices: {data_dict['val_indices'][0]}-{data_dict['val_indices'][-1]}\n")
            f.write(f"Test indices: {data_dict['test_indices'][0]}-{data_dict['test_indices'][-1]}\n")
            f.write(f"Original number of stocks: {data_dict['dataset_metadata']['original_num_stocks']}\n")
            f.write(f"Removed stocks: {data_dict['dataset_metadata']['removed_stocks']}\n")
            f.write(f"Min data threshold: {data_dict['dataset_metadata']['min_data_threshold']}\n")
            f.write(f"Imputation method: {data_dict['dataset_metadata']['impute_method']}\n")
        
        logger.info(f"Processed data saved to {output_dir}") 