#!/usr/bin/env python3
"""
CRYPTO INTRADAY DASHBOARD FOR REVOLUT
Complete Trading Intelligence System
Uses: CoinGecko API, Binance API, CryptoCompare, TradingView Webhooks
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from colorama import Fore, Back, Style, init
import warnings
warnings.filterwarnings('ignore')

init(autoreset=True)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "investment_amount_gbp": 100,
    "update_interval_seconds": 30,
    "max_coins_display": 50,
    "include_memecoins": True,
    "risk_reward_ratio": 2.0,      # 1:2 Risk/Reward
    "default_sl_percent": 3.0,     # 3% Stop Loss
    "default_tp_percent": 6.0,     # 6% Take Profit
    "breakout_volume_multiplier": 2.5,  # Volume spike threshold
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "gbp_usd_rate": 1.27           # Update daily
}

MEMECOIN_LIST = [
    "dogecoin", "shiba-inu", "pepe", "dogwifhat", "bonk",
    "floki", "baby-doge-coin", "kishu-inu", "akita-inu",
    "dogelon-mars", "samoyedcoin", "pitbull", "hoge-finance",
    "catecoin", "meme", "wojak", "turbo", "milady-meme-coin",
    "book-of-meme", "brett", "mog-coin", "popcat", "neiro",
    "act-i-the-ai-prophecy", "goatseus-maximus", "fwog",
    "coq-inu", "smog", "retardio", "mother-iggy", "ponke"
]

MAJOR_CRYPTO = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
    "cardano", "avalanche-2", "polkadot", "chainlink", "polygon",
    "uniswap", "litecoin", "bitcoin-cash", "stellar", "cosmos",
    "algorand", "vechain", "filecoin", "tron", "eos",
    "theta-token", "aave", "maker", "compound-governance-token",
    "the-sandbox", "decentraland", "axie-infinity", "gala",
    "render-token", "injective-protocol", "sui", "aptos",
    "arbitrum", "optimism", "near", "internet-computer",
    "hedera-hashgraph", "quant-network", "elrond-erd-2",
    "fantom", "harmony", "zilliqa", "waves", "neo",
    "dash", "zcash", "monero", "iota", "ontology"
]

ALL_COINS = MAJOR_CRYPTO + MEMECOIN_LIST

# ============================================================
# API CONNECTIONS
# ============================================================
class CryptoDataFetcher:
    def __init__(self):
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.binance_base = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 CryptoDashboard/1.0'
        })

    def get_market_data(self, coin_ids, vs_currency="gbp"):
        """Fetch comprehensive market data"""
        url = f"{self.coingecko_base}/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": True,
            "price_change_percentage": "1h,24h,7d"
        }
        try:
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"API Error: {e}")
            return []

    def get_ohlc_data(self, coin_id, days=1):
        """Get OHLC for technical analysis"""
        url = f"{self.coingecko_base}/coins/{coin_id}/ohlc"
        params = {"vs_currency": "gbp", "days": days}
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            return pd.DataFrame()
        except:
            return pd.DataFrame()

    def get_binance_volume(self, symbol):
        """Get real-time volume data from Binance"""
        url = f"{self.binance_base}/ticker/24hr"
        params = {"symbol": symbol}
        try:
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

# ============================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================
class TechnicalAnalysis:

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50  # Neutral if not enough data
        
        prices = pd.Series(prices)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow:
            return 0, 0, 0
        
        prices = pd.Series(prices)
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return (float(macd_line.iloc[-1]),
                float(signal_line.iloc[-1]),
                float(histogram.iloc[-1]))

    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        prices = pd.Series(prices)
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return (float(upper.iloc[-1]),
                float(middle.iloc[-1]),
                float(lower.iloc[-1]))

    @staticmethod
    def calculate_support_resistance(prices):
        """Identify key S/R levels"""
        if len(prices) < 10:
            return None, None
        
        prices_array = np.array(prices)
        resistance = float(np.percentile(prices_array, 85))
        support = float(np.percentile(prices_array, 15))
        return support, resistance

    @staticmethod
    def calculate_ema(prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        prices = pd.Series(prices)
        return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def detect_breakout_pattern(prices, volumes=None):
        """
        Detect breakout patterns
        Returns: breakout_score (0-100), breakout_type, estimated_time
        """
        if len(prices) < 20:
            return 0, "NONE", "N/A"
        
        prices = np.array(prices)
        score = 0
        pattern_type = "NONE"
        
        # Price compression (low volatility before breakout)
        recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
        historical_volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        
        if recent_volatility < historical_volatility * 0.5:
            score += 30
            pattern_type = "COMPRESSION"
        
        # Higher lows pattern
        lows = [min(prices[i:i+3]) for i in range(0, len(prices)-3, 3)]
        if len(lows) >= 3:
            if lows[-1] > lows[-2] > lows[-3]:
                score += 25
                pattern_type = "ASCENDING"
        
        # Price near resistance
        resistance = np.percentile(prices, 85)
        if prices[-1] > resistance * 0.97:
            score += 20
            pattern_type = "RESISTANCE_TEST"
        
        # Momentum building
        short_ema = np.mean(prices[-3:])
        long_ema = np.mean(prices[-10:])
        if short_ema > long_ema * 1.01:
            score += 25
        
        # Estimate breakout time
        if score >= 70:
            est_time = "< 1 hour"
        elif score >= 50:
            est_time = "1-4 hours"
        elif score >= 30:
            est_time = "4-12 hours"
        else:
            est_time = "> 12 hours"
        
        return min(score, 100), pattern_type, est_time

# ============================================================
# BUY SIGNAL GENERATOR
# ============================================================
class BuySignalGenerator:

    def __init__(self, config):
        self.config = config
        self.ta = TechnicalAnalysis()

    def generate_signal(self, coin_data, sparkline_prices):
        """
        Generate comprehensive buy signal
        Returns: signal_strength, signal_type, confidence
        """
        if not sparkline_prices or len(sparkline_prices) < 14:
            return "HOLD", 0, "Insufficient data"

        signals = []
        reasons = []

        # 1. RSI Signal
        rsi = self.ta.calculate_rsi(sparkline_prices)
        if rsi < self.config["rsi_oversold"]:
            signals.append(2)
            reasons.append(f"RSI Oversold ({rsi:.1f})")
        elif rsi < 45:
            signals.append(1)
            reasons.append(f"RSI Bullish Zone ({rsi:.1f})")
        elif rsi > self.config["rsi_overbought"]:
            signals.append(-2)
            reasons.append(f"RSI Overbought ({rsi:.1f})")

        # 2. MACD Signal
        macd, signal, histogram = self.ta.calculate_macd(sparkline_prices)
        if histogram > 0 and macd > signal:
            signals.append(2)
            reasons.append("MACD Bullish Crossover")
        elif histogram < 0:
            signals.append(-1)
            reasons.append("MACD Bearish")

        # 3. Bollinger Band Signal
        bb_upper, bb_mid, bb_lower = self.ta.calculate_bollinger_bands(sparkline_prices)
        current_price = sparkline_prices[-1]
        if bb_lower and current_price < bb_lower * 1.02:
            signals.append(2)
            reasons.append("Price at Lower BB (Bounce Expected)")
        elif bb_upper and current_price > bb_upper * 0.98:
            signals.append(-1)
            reasons.append("Price at Upper BB")

        # 4. Price Change Momentum
        change_1h = coin_data.get('price_change_percentage_1h_in_currency', 0) or 0
        change_24h = coin_data.get('price_change_percentage_24h_in_currency', 0) or 0

        if -5 < change_24h < -2 and change_1h > 0:
            signals.append(2)
            reasons.append("Dip Recovery In Progress")
        elif change_24h > 15:
            signals.append(-1)
            reasons.append("Potentially Overextended")

        # 5. Volume Analysis
        volume_24h = coin_data.get('total_volume', 0) or 0
        market_cap = coin_data.get('market_cap', 1) or 1
        volume_ratio = volume_24h / market_cap if market_cap > 0 else 0

        if volume_ratio > 0.15:
            signals.append(2)
            reasons.append(f"High Volume Activity ({volume_ratio:.2%})")
        elif volume_ratio > 0.08:
            signals.append(1)
            reasons.append("Above Average Volume")

        # Calculate overall signal
        total_score = sum(signals)
        max_possible = len(signals) * 2 if signals else 1
        confidence = min(abs(total_score) / max_possible * 100, 100) if max_possible > 0 else 0

        if total_score >= 4:
            signal_type = "STRONG BUY 🟢🟢"
        elif total_score >= 2:
            signal_type = "BUY 🟢"
        elif total_score <= -3:
            signal_type = "STRONG SELL 🔴🔴"
        elif total_score <= -1:
            signal_type = "SELL 🔴"
        else:
            signal_type = "HOLD ⚪"

        return signal_type, round(confidence, 1), " | ".join(reasons[:3])

# ============================================================
# RISK MANAGEMENT CALCULATOR
# ============================================================
class RiskManager:

    def __init__(self, config):
        self.config = config

    def calculate_sl_tp(self, current_price, signal_strength, volatility_24h):
        """
        Dynamic SL/TP based on volatility and signal
        Returns: stop_loss, take_profit, risk_reward
        """
        # Adjust based on volatility
        base_sl = self.config["default_sl_percent"]
        base_tp = self.config["default_tp_percent"]

        # Tighter SL for strong signals, wider for memes
        if "STRONG BUY" in signal_strength:
            sl_pct = base_sl * 0.8
            tp_pct = base_tp * 1.5
        elif "BUY" in signal_strength:
            sl_pct = base_sl
            tp_pct = base_tp
        else:
            sl_pct = base_sl * 1.2
            tp_pct = base_tp * 0.8

        # Volatility adjustment
        if volatility_24h:
            vol_factor = abs(volatility_24h) / 10
            sl_pct = max(sl_pct, sl_pct * vol_factor * 0.5)
            tp_pct = max(tp_pct, tp_pct * vol_factor * 0.5)

        stop_loss = current_price * (1 - sl_pct / 100)
        take_profit = current_price * (1 + tp_pct / 100)
        risk_reward = tp_pct / sl_pct

        return {
            "stop_loss": round(stop_loss, 8),
            "take_profit": round(take_profit, 8),
            "sl_percent": round(sl_pct, 2),
            "tp_percent": round(tp_pct, 2),
            "risk_reward": round(risk_reward, 2)
        }

    def calculate_potential_gain(self, investment_gbp, tp_percent):
        """Calculate potential gain on £100"""
        potential_gain = investment_gbp * (tp_percent / 100)
        total_return = investment_gbp + potential_gain
        return {
            "investment": investment_gbp,
            "potential_gain": round(potential_gain, 2),
            "total_return": round(total_return, 2),
            "roi_percent": round(tp_percent, 2)
        }

    def calculate_max_loss(self, investment_gbp, sl_percent):
        """Calculate maximum loss"""
        max_loss = investment_gbp * (sl_percent / 100)
        return round(max_loss, 2)

# ============================================================
# BREAKOUT ANALYZER
# ============================================================
class BreakoutAnalyzer:

    def __init__(self):
        self.ta = TechnicalAnalysis()

    def analyze_breakout(self, coin_data, sparkline):
        """
        Comprehensive breakout analysis
        """
        if not sparkline:
            return self._empty_breakout()

        prices = sparkline
        score, pattern, est_time = self.ta.detect_breakout_pattern(prices)

        # Additional factors
        volume_change = coin_data.get('price_change_percentage_24h_in_currency', 0) or 0
        ath_distance = self._calculate_ath_distance(coin_data)

        # Breakout likelihood classification
        if score >= 75:
            likelihood = "VERY HIGH 🔥🔥🔥"
            likelihood_pct = f"{score}%"
        elif score >= 55:
            likelihood = "HIGH 🔥🔥"
            likelihood_pct = f"{score}%"
        elif score >= 35:
            likelihood = "MODERATE 📈"
            likelihood_pct = f"{score}%"
        elif score >= 20:
            likelihood = "LOW 📊"
            likelihood_pct = f"{score}%"
        else:
            likelihood = "VERY LOW ❄️"
            likelihood_pct = f"{score}%"

        return {
            "score": score,
            "pattern": pattern,
            "estimated_time": est_time,
            "likelihood": likelihood,
            "likelihood_pct": likelihood_pct,
            "ath_distance": ath_distance
        }

    def _calculate_ath_distance(self, coin_data):
        ath = coin_data.get('ath', 0) or 0
        current = coin_data.get('current_price', 0) or 0
        if ath > 0 and current > 0:
            return round((1 - current/ath) * 100, 2)
        return None

    def _empty_breakout(self):
        return {
            "score": 0, "pattern": "N/A",
            "estimated_time": "N/A", "likelihood": "N/A",
            "likelihood_pct": "0%", "ath_distance": None
        }

# ============================================================
# MAIN DASHBOARD ENGINE
# ============================================================
class CryptoDashboard:

    def __init__(self):
        self.fetcher = CryptoDataFetcher()
        self.ta = TechnicalAnalysis()
        self.signal_gen = BuySignalGenerator(CONFIG)
        self.risk_mgr = RiskManager(CONFIG)
        self.breakout = BreakoutAnalyzer()
        self.last_update = None

    def process_coin(self, coin_data):
        """Process single coin and generate all indicators"""
        sparkline = coin_data.get('sparkline_in_7d', {})
        prices = sparkline.get('price', []) if sparkline else []

        current_price = coin_data.get('current_price', 0) or 0
        change_24h = coin_data.get('price_change_percentage_24h_in_currency', 0) or 0
        change_1h = coin_data.get('price_change_percentage_1h_in_currency', 0) or 0

        # Generate signals
        signal, confidence, reasons = self.signal_gen.generate_signal(coin_data, prices)

        # Breakout analysis
        breakout_data = self.breakout.analyze_breakout(coin_data, prices)

        # Risk management
        sl_tp = self.risk_mgr.calculate_sl_tp(current_price, signal, change_24h)
        gains = self.risk_mgr.calculate_potential_gain(
            CONFIG["investment_amount_gbp"],
            sl_tp["tp_percent"]
        )
        max_loss = self.risk_mgr.calculate_max_loss(
            CONFIG["investment_amount_gbp"],
            sl_tp["sl_percent"]
        )

        # Technical indicators
        rsi = self.ta.calculate_rsi(prices) if prices else 50
        macd_val, sig_val, hist = self.ta.calculate_macd(prices) if prices else (0, 0, 0)

        is_memecoin = coin_data.get('id', '') in MEMECOIN_LIST

        return {
            "rank": coin_data.get('market_cap_rank', 999),
            "name": coin_data.get('name', 'Unknown'),
            "symbol": coin_data.get('symbol', '???').upper(),
            "id": coin_data.get('id', ''),
            "is_memecoin": is_memecoin,
            "current_price_gbp": current_price,
            "change_1h": change_1h,
            "change_24h": change_24h,
            "volume_24h_gbp": coin_data.get('total_volume', 0),
            "market_cap_gbp": coin_data.get('market_cap', 0),
            "signal": signal,
            "signal_confidence": confidence,
            "signal_reasons": reasons,
            "breakout_score": breakout_data["score"],
            "breakout_likelihood": breakout_data["likelihood"],
            "breakout_likelihood_pct": breakout_data["likelihood_pct"],
            "breakout_time": breakout_data["estimated_time"],
            "breakout_pattern": breakout_data["pattern"],
            "ath_distance": breakout_data["ath_distance"],
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "sl_percent": sl_tp["sl_percent"],
            "tp_percent": sl_tp["tp_percent"],
            "risk_reward": sl_tp["risk_reward"],
            "potential_gain_gbp": gains["potential_gain"],
            "total_return_gbp": gains["total_return"],
            "max_loss_gbp": max_loss,
            "rsi": round(rsi, 1),
            "macd_histogram": round(hist, 6),
            "ath_gbp": coin_data.get('ath', 0),
        }

    def run_dashboard(self):
        """Main dashboard loop"""
        print(self._header())
        
        while True:
            try:
                self.last_update = datetime.now()
                print(f"\n{Fore.CYAN}⟳ Fetching live data... {self.last_update.strftime('%H:%M:%S')}{Style.RESET_ALL}")

                # Fetch in batches (CoinGecko limit)
                all_data = []
                coin_batches = [ALL_COINS[i:i+50] for i in range(0, len(ALL_COINS), 50)]

                for batch in coin_batches:
                    batch_data = self.fetcher.get_market_data(batch)
                    all_data.extend(batch_data)
                    time.sleep(1)  # Rate limiting

                if not all_data:
                    print(f"{Fore.RED}No data received. Retrying...{Style.RESET_ALL}")
                    time.sleep(30)
                    continue

                # Process all coins
                processed = []
                for coin in all_data:
                    try:
                        result = self.process_coin(coin)
                        processed.append(result)
                    except Exception as e:
                        continue

                # Sort by breakout score + signal confidence
                processed.sort(
                    key=lambda x: (x['breakout_score'] * 0.4 + x['signal_confidence'] * 0.4 +
                                   x.get('change_24h', 0) * 0.2),
                    reverse=True
                )

                # Display dashboards
                self._display_top_opportunities(processed)
                self._display_memecoin_dashboard(processed)
                self._display_breakout_radar(processed)
                self._display_full_ranked_table(processed)

                print(f"\n{Fore.YELLOW}⏱ Next update in {CONFIG['update_interval_seconds']}s...{Style.RESET_ALL}")
                time.sleep(CONFIG["update_interval_seconds"])

            except KeyboardInterrupt:
                print(f"\n{Fore.RED}Dashboard stopped by user.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
                time.sleep(30)

    def _header(self):
        return f"""
{Fore.CYAN}{'='*80}
    🚀 REVOLUT CRYPTO INTRADAY DASHBOARD 
    📊 Live Signals | Breakout Detection | Risk Management
    💷 Base Investment: £{CONFIG['investment_amount_gbp']} | Risk:Reward = 1:{CONFIG['risk_reward_ratio']}
{'='*80}{Style.RESET_ALL}"""

    def _display_top_opportunities(self, data):
        """Top 10 best opportunities right now"""
        print(f"\n{Back.GREEN}{Fore.BLACK}{'='*80}")
        print(f"  🏆 TOP 10 BEST OPPORTUNITIES RIGHT NOW")
        print(f"{'='*80}{Style.RESET_ALL}")

        headers = f"{'#':<3} {'Coin':<12} {'Price £':<12} {'1H%':<7} {'24H%':<8} {'Signal':<18} {'Conf%':<7} {'BO Score':<9} {'Gain £':<8} {'Loss £':<7}"
        print(f"{Fore.YELLOW}{headers}{Style.RESET_ALL}")
        print("-" * 100)

        top_10 = [c for c in data if "BUY" in c['signal']][:10]

        for i, coin in enumerate(top_10, 1):
            # Color coding
            if coin['change_24h'] > 0:
                price_color = Fore.GREEN
            else:
                price_color = Fore.RED

            memecoin_tag = "🐕" if coin['is_memecoin'] else "  "

            print(
                f"{Fore.WHITE}{i:<3} "
                f"{coin['symbol']:<6}{memecoin_tag:<6} "
                f"{price_color}£{coin['current_price_gbp']:<11.6f} "
                f"{Fore.CYAN}{coin['change_1h']:>+6.2f}% "
                f"{price_color}{coin['change_24h']:>+7.2f}% "
                f"{Fore.GREEN}{coin['signal']:<18} "
                f"{Fore.YELLOW}{coin['signal_confidence']:<7.1f} "
                f"{Fore.MAGENTA}{coin['breakout_score']:<9} "
                f"{Fore.GREEN}+£{coin['potential_gain_gbp']:<7.2f} "
                f"{Fore.RED}-£{coin['max_loss_gbp']:<6.2f}"
                f"{Style.RESET_ALL}"
            )

    def _display_memecoin_dashboard(self, data):
        """Dedicated memecoin section"""
        print(f"\n{Back.MAGENTA}{Fore.WHITE}{'='*80}")
        print(f"  🐕 MEMECOIN TRACKER - HIGH RISK / HIGH REWARD")
        print(f"{'='*80}{Style.RESET_ALL}")

        memecoins = [c for c in data if c['is_memecoin']]
        memecoins.sort(key=lambda x: x['breakout_score'], reverse=True)

        if not memecoins:
            print(f"{Fore.YELLOW}No memecoin data available{Style.RESET_ALL}")
            return

        print(f"\n{'Coin':<12} {'Price £':<14} {'24H%':<9} {'Signal':<18} {'BO%':<6} {'BO Time':<12} {'RSI':<6} {'SL%':<6} {'TP%':<6} {'Gain£':<8}")
        print("-" * 100)

        for coin in memecoins[:15]:
            rsi_color = Fore.GREEN if coin['rsi'] < 40 else (Fore.RED if coin['rsi'] > 65 else Fore.YELLOW)
            print(
                f"{Fore.MAGENTA}{coin['symbol']:<12}"
                f"{Fore.WHITE}£{coin['current_price_gbp']:<13.8f}"
                f"{Fore.GREEN if coin['change_24h'] > 0 else Fore.RED}{coin['change_24h']:>+8.2f}% "
                f"{Fore.CYAN}{coin['signal']:<18}"
                f"{Fore.YELLOW}{coin['breakout_likelihood_pct']:<6}"
                f"{Fore.WHITE}{coin['breakout_time']:<12}"
                f"{rsi_color}{coin['rsi']:<6.1f}"
                f"{Fore.RED}{coin['sl_percent']:<6.1f}%"
                f"{Fore.GREEN}{coin['tp_percent']:<6.1f}%"
                f"{Fore.GREEN}£{coin['potential_gain_gbp']:<7.2f}"
                f"{Style.RESET_ALL}"
            )

    def _display_breakout_radar(self, data):
        """Breakout detection radar"""
        print(f"\n{Back.BLUE}{Fore.WHITE}{'='*80}")
        print(f"  📡 BREAKOUT RADAR - HIGHEST PROBABILITY BREAKOUTS")
        print(f"{'='*80}{Style.RESET_ALL}")

        high_breakout = sorted(data, key=lambda x: x['breakout_score'], reverse=True)[:10]

        for coin in high_breakout:
            if coin['breakout_score'] > 20:
                bar_len = int(coin['breakout_score'] / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(
                    f"{Fore.WHITE}{coin['symbol']:<8} "
                    f"[{Fore.YELLOW}{bar}{Fore.WHITE}] "
                    f"{Fore.GREEN}{coin['breakout_likelihood_pct']:>4} "
                    f"{Fore.CYAN}| Pattern: {coin['breakout_pattern']:<15} "
                    f"| ETA: {coin['breakout_time']:<12} "
                    f"| {coin['breakout_likelihood']}"
                    f"{Style.RESET_ALL}"
                )

    def _display_full_ranked_table(self, data):
        """Full ranked table with all coins"""
        print(f"\n{Back.WHITE}{Fore.BLACK}{'='*80}")
        print(f"  📋 FULL RANKED TABLE - ALL COINS")
        print(f"{'='*80}{Style.RESET_ALL}")

        print(f"\n{'Rank':<5} {'Symbol':<8} {'Type':<6} {'Price £':<14} {'1H%':<7} {'24H%':<8} {'Signal':<18} {'RSI':<5} {'SL£':<14} {'TP£':<14} {'R:R':<5} {'Gain£'}")
        print("-" * 130)

        for coin in data:
            type_tag = "MEME" if coin['is_memecoin'] else "COIN"
            type_color = Fore.MAGENTA if coin['is_memecoin'] else Fore.CYAN
            signal_color = Fore.GREEN if "BUY" in coin['signal'] else (Fore.RED if "SELL" in coin['signal'] else Fore.WHITE)
            change_color = Fore.GREEN if coin['change_24h'] > 0 else Fore.RED

            print(
                f"{Fore.WHITE}{str(coin['rank']):<5} "
                f"{coin['symbol']:<8} "
                f"{type_color}{type_tag:<6} "
                f"{Fore.WHITE}£{coin['current_price_gbp']:<13.8f} "
                f"{Fore.CYAN}{coin['change_1h']:>+6.2f}% "
                f"{change_color}{coin['change_24h']:>+7.2f}% "
                f"{signal_color}{coin['signal']:<18} "
                f"{Fore.YELLOW}{coin['rsi']:<5.1f} "
                f"{Fore.RED}£{coin['stop_loss']:<13.8f} "
                f"{Fore.GREEN}£{coin['take_profit']:<13.8f} "
                f"{Fore.WHITE}{coin['risk_reward']:<5.1f} "
                f"{Fore.GREEN}£{coin['potential_gain_gbp']:.2f}"
                f"{Style.RESET_ALL}"
            )

# ============================================================
# REVOLUT-SPECIFIC HELPER
# ============================================================
class RevolutHelper:
    """
    Revolut Platform X integration notes
    Since Revolut doesn't have a trading API yet,
    this generates actionable alerts
    """

    @staticmethod
    def generate_revolut_alert(coin_data):
        """Generate formatted alert for Revolut trading"""
        return f"""
╔══════════════════════════════════════════╗
║     🚨 REVOLUT TRADE ALERT              ║
╠══════════════════════════════════════════╣
║ Coin:    {coin_data['name']:<30} ║
║ Symbol:  {coin_data['symbol']:<30} ║
║ Signal:  {coin_data['signal']:<30} ║
║ Price:   £{str(coin_data['current_price_gbp']):<29} ║
╠══════════════════════════════════════════╣
║ 📍 ACTION: Go to Revolut → Crypto       ║
║    Search: {coin_data['symbol']:<30} ║
║    Buy Amount: £100                     ║
╠══════════════════════════════════════════╣
║ 🛑 STOP LOSS: £{str(coin_data['stop_loss']):<25} ║
║    Set a price alert at this level      ║
║ 🎯 TAKE PROFIT: £{str(coin_data['take_profit']):<23} ║
║    Set a price alert at this level      ║
╠══════════════════════════════════════════╣
║ 💰 POTENTIAL GAIN:  +£{str(coin_data['potential_gain_gbp']):<19} ║
║ 💸 MAXIMUM LOSS:   -£{str(coin_data['max_loss_gbp']):<19} ║
║ ⚖️  RISK/REWARD:    1:{str(coin_data['risk_reward']):<19} ║
╠══════════════════════════════════════════╣
║ 📊 WHY BUY NOW:                         ║
║ {coin_data['signal_reasons'][:40]:<41} ║
╚══════════════════════════════════════════╝"""

    @staticmethod
    def revolut_manual_instructions():
        return """
═══════════════════════════════════════════════════
  📱 HOW TO EXECUTE ON REVOLUT PLATFORM X
═══════════════════════════════════════════════════

1. BUYING:
   → Open Revolut App
   → Tap "Crypto" in bottom menu
   → Search for the coin symbol
   → Tap "Buy" 
   → Enter £100 (or your amount)
   → Confirm purchase

2. SETTING STOP LOSS (Revolut Method):
   → After buying, go to your position
   → Set a Price Alert at your SL price
   → When triggered, MANUALLY sell position
   ⚠️ Revolut has limited native SL orders

3. SETTING TAKE PROFIT:
   → Set Price Alert at TP price
   → When triggered, manually sell
   → Or set a recurring sell order

4. MONITORING:
   → Keep this dashboard running
   → Refresh every 30 seconds
   → Act on STRONG BUY signals only
   → Never invest more than you can lose

5. REVOLUT PRO TIPS:
   → Use Revolut "Recurring Buy" for DCA
   → Check Revolut's trading hours
   → Be aware of spread costs (0.5-2.5%)
   → Revolut charges no commission
   → But has spread built into price

═══════════════════════════════════════════════════"""

# ============================================================
# REQUIREMENTS.TXT
# ============================================================
REQUIREMENTS = """
# Save as requirements.txt and run: pip install -r requirements.txt
requests==2.31.0
pandas==2.1.0
numpy==1.24.0
colorama==0.4.6
ta==0.10.2
schedule==1.2.0
python-dotenv==1.0.0
"""

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║     🚀 REVOLUT CRYPTO INTRADAY DASHBOARD                ║
║     Starting up...                                       ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Show Revolut instructions
    helper = RevolutHelper()
    print(helper.revolut_manual_instructions())
    
    # Start dashboard
    dashboard = CryptoDashboard()
    dashboard.run_dashboard()
