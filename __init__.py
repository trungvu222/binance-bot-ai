"""
Binance Futures Trading Bot
Bot AI trading futures với độ chính xác cao
"""

__version__ = "1.0.0"
__author__ = "AI Trading Team"
__description__ = "Professional AI-powered Binance futures trading bot"

from .main import BinanceFuturesBot
from .binance_client import BinanceFuturesClient
from .technical_analysis import TechnicalAnalyzer
from .ai_engine import AITradingEngine
from .risk_management import RiskManager
from .backtest import BacktestEngine
from .config import Config

__all__ = [
    'BinanceFuturesBot',
    'BinanceFuturesClient', 
    'TechnicalAnalyzer',
    'AITradingEngine',
    'RiskManager',
    'BacktestEngine',
    'Config'
]
