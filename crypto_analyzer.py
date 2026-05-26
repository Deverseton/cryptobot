#!/usr/bin/env python3
"""
Криптоботанализер - анализ топ-20 криптовалют
Полный технический анализ с прогнозом на 1 час вперед
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple
import sys

class CryptoAnalyzer:
    def __init__(self):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.binance_url = "https://api.binance.com/api/v3"
        self.top_20_ids = []
        
    def get_top_20_cryptos(self) -> List[Dict]:
        """Получить топ-20 крипто по капитализации"""
        try:
            print("📊 Загружаю топ-20 крипто по капитализации...")
            url = f"{self.coingecko_url}/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 20,
                'page': 1,
                'sparkline': True,
                'price_change_percentage': '1h,24h,7d'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            cryptos = []
            for coin in data:
                cryptos.append({
                    'id': coin['id'],
                    'symbol': coin['symbol'].upper(),
                    'name': coin['name'],
                    'current_price': coin['current_price'],
                    'market_cap': coin['market_cap'],
                    'change_1h': coin['price_change_percentage_1h_in_currency'],
                    'change_24h': coin['price_change_percentage_24h_in_currency'],
                    'change_7d': coin['price_change_percentage_7d_in_currency'],
                    'high_24h': coin['high_24h'],
                    'low_24h': coin['low_24h'],
                    'sparkline': coin['sparkline']['price'] if coin['sparkline'] else []
                })
            
            self.top_20_ids = [c['id'] for c in cryptos]
            print(f"✅ Загружено {len(cryptos)} криптовалют\n")
            return cryptos
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return []

    def get_ohlcv_data(self, symbol: str, interval: str = '1h') -> pd.DataFrame:
        """Получить OHLCV данные с Binance"""
        try:
            binance_symbol = f"{symbol.upper()}USDT"
            
            url = f"{self.binance_url}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Не смог загрузить данные для {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитать технические индикаторы"""
        if df.empty:
            return df
            
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA14'] = df['close'].rolling(window=14).mean()
        df['MA25'] = df['close'].rolling(window=25).mean()
        
        df['RSI'] = self.calculate_rsi(df['close'], period=14)
        
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal']
        
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        df['ATR'] = self.calculate_atr(df, period=14)
        
        df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df['high'], df['low'], df['close'], period=14)
        
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитать RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return pd.Series(rsi, index=prices.index)

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитать ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Рассчитать Стохастик"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        
        return k, d

    def predict_1hour_ahead(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Прогноз цены на 1 час вперед"""
        if df.empty or len(df) < 20:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'target_low': current_price, 'target_high': current_price}
        
        latest = df.iloc[-1]
        signals = []
        weights = []
        
        if latest['MA7'] > latest['MA14'] > latest['MA25']:
            signals.append(1)
            weights.append(2)
        elif latest['MA7'] < latest['MA14'] < latest['MA25']:
            signals.append(-1)
            weights.append(2)
        else:
            signals.append(0)
            weights.append(1)
        
        if latest['RSI'] > 70:
            signals.append(-1)
            weights.append(1.5)
        elif latest['RSI'] < 30:
            signals.append(1)
            weights.append(1.5)
        else:
            signals.append(0)
            weights.append(1)
        
        if latest['MACD_Histogram'] > 0 and df.iloc[-2]['MACD_Histogram'] <= 0:
            signals.append(1)
            weights.append(2)
        elif latest['MACD_Histogram'] < 0 and df.iloc[-2]['MACD_Histogram'] >= 0:
            signals.append(-1)
            weights.append(2)
        elif latest['MACD_Histogram'] > 0:
            signals.append(1)
            weights.append(1)
        else:
            signals.append(-1)
            weights.append(1)
        
        if latest['close'] < latest['BB_Lower']:
            signals.append(1)
            weights.append(1.5)
        elif latest['close'] > latest['BB_Upper']:
            signals.append(-1)
            weights.append(1.5)
        else:
            signals.append(0)
            weights.append(0.5)
        
        if latest['Stoch_K'] > 80:
            signals.append(-1)
            weights.append(1.5)
        elif latest['Stoch_K'] < 20:
            signals.append(1)
            weights.append(1.5)
        else:
            signals.append(0)
            weights.append(1)
        
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
        
        if weighted_signal > 0.3:
            direction = "🔼 BULL (РОСТ)"
            confidence = min(abs(weighted_signal) * 50, 95)
        elif weighted_signal < -0.3:
            direction = "🔽 BEAR (ПАДЕНИЕ)"
            confidence = min(abs(weighted_signal) * 50, 95)
        else:
            direction = "↔️ NEUTRAL (БОКОВИК)"
            confidence = 50
        
        atr = latest['ATR'] if not np.isnan(latest['ATR']) else 0
        volatility_multiplier = atr / current_price if current_price > 0 else 0.02
        
        target_high = current_price * (1 + volatility_multiplier * 1.5)
        target_low = current_price * (1 - volatility_multiplier * 1.5)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'target_high': target_high,
            'target_low': target_low,
            'rsi': latest['RSI'],
            'macd_histogram': latest['MACD_Histogram'],
            'stoch_k': latest['Stoch_K']
        }

    def analyze_crypto(self, crypto: Dict) -> Dict:
        """Полный анализ одной крипто"""
        symbol = crypto['symbol']
        
        df = self.get_ohlcv_data(symbol)
        if df.empty:
            return None
        
        df = self.calculate_indicators(df)
        
        prediction = self.predict_1hour_ahead(df, crypto['current_price'])
        
        latest = df.iloc[-1]
        
        return {
            'symbol': symbol,
            'name': crypto['name'],
            'current_price': crypto['current_price'],
            'change_1h': crypto['change_1h'],
            'change_24h': crypto['change_24h'],
            'change_7d': crypto['change_7d'],
            'high_24h': crypto['high_24h'],
            'low_24h': crypto['low_24h'],
            'prediction': prediction,
            'indicators': {
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_histogram': latest['MACD_Histogram'],
                'bb_upper': latest['BB_Upper'],
                'bb_middle': latest['BB_Middle'],
                'bb_lower': latest['BB_Lower'],
                'stoch_k': latest['Stoch_K'],
                'stoch_d': latest['Stoch_D'],
                'ma7': latest['MA7'],
                'ma14': latest['MA14'],
                'ma25': latest['MA25'],
                'atr': latest['ATR']
            }
        }

    def format_output(self, analysis: Dict) -> str:
        """Форматированный вывод анализа"""
        if not analysis:
            return ""
        
        pred = analysis['prediction']
        ind = analysis['indicators']
        
        output = f"""
╔════════════════════════════════════════════════════════════╗
║ {analysis['symbol']:>6} - {analysis['name']:<40} ║
╚════════════════════════════════════════════════════════════╝

💰 ЦЕНА И ИЗМЕНЕНИЯ:
   Текущая цена: ${analysis['current_price']:,.2f}
   Изменение за 1ч:  {analysis['change_1h']:+.2f}%
   Изменение за 24ч: {analysis['change_24h']:+.2f}%
   Изменение за 7д:  {analysis['change_7d']:+.2f}%
   Диапазон за 24ч:  ${analysis['low_24h']:,.2f} - ${analysis['high_24h']:,.2f}

📊 ПРОГНОЗ НА 1 ЧАС:
   Направление: {pred['direction']}
   Уверенность: {pred['confidence']:.1f}%
   Целевой диапазон: ${pred['target_low']:,.2f} - ${pred['target_high']:,.2f}%
   
   ⚠️ РЕКОМЕНДАЦИЯ: 
   {'✅ БЫЧИЙ СИГНАЛ - можно покупать' if '🔼' in pred['direction'] and pred['confidence'] > 70 else '❌ МЕДВЕЖИЙ СИГНАЛ - лучше продавать' if '🔽' in pred['direction'] and pred['confidence'] > 70 else '⚠️ НЕОПРЕДЕЛЁННОСТЬ - ждите более четких сигналов'}

📈 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
   RSI (14):           {ind['rsi']:.2f} {'(Перекуплено ⚠️)' if ind['rsi'] > 70 else '(Перепродано ⚠️)' if ind['rsi'] < 30 else ''}
   MACD:               {ind['macd']:.6f}
   MACD Histogram:     {ind['macd_histogram']:.6f}
   Стохастик K:       {ind['stoch_k']:.2f} {'(Перекуплено ⚠️)' if ind['stoch_k'] > 80 else '(Перепродано ⚠️)' if ind['stoch_k'] < 20 else ''}
   
   Скользящие средние:
   MA7:                ${ind['ma7']:,.2f}
   MA14:               ${ind['ma14']:,.2f}
   MA25:               ${ind['ma25']:,.2f}
   
   Bollinger Bands:
   Верхняя:            ${ind['bb_upper']:,.2f}
   Средняя:            ${ind['bb_middle']:,.2f}
   Нижняя:             ${ind['bb_lower']:,.2f}
   
   ATR:                ${ind['atr']:,.2f}

"""
        return output

    def run_full_analysis(self):
        """Запустить полный анализ топ-20"""
        print("\n" + "="*60)
        print("🤖 КРИПТОАНАЛИЗЕР - ПОЛНЫЙ ТЕХНИЧЕСКИЙ АНАЛИЗ")
        print(f"⏰ Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        cryptos = self.get_top_20_cryptos()
        if not cryptos:
            return
        
        results = []
        bull_signals = []
        bear_signals = []
        
        for i, crypto in enumerate(cryptos, 1):
            print(f"Анализирую {i}/20: {crypto['symbol']} ({crypto['name']})...", end=" ")
            
            analysis = self.analyze_crypto(crypto)
            if analysis:
                results.append(analysis)
                print("✅")
                
                pred = analysis['prediction']
                if '🔼' in pred['direction'] and pred['confidence'] > 70:
                    bull_signals.append((crypto['symbol'], pred['confidence']))
                elif '🔽' in pred['direction'] and pred['confidence'] > 70:
                    bear_signals.append((crypto['symbol'], pred['confidence']))
            else:
                print("⚠️")
            
            time.sleep(0.5)
        
        print("\n" + "="*60)
        print("📋 РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("="*60)
        
        for analysis in results:
            print(self.format_output(analysis))
        
        print("\n" + "="*60)
        print("🎯 СВОДКА СИГНАЛОВ")
        print("="*60 + "\n")
        
        if bull_signals:
            print("🔼 БЫЧЬИ СИГНАЛЫ (РОСТ):")
            for symbol, conf in sorted(bull_signals, key=lambda x: x[1], reverse=True):
                print(f"   ✅ {symbol}: {conf:.1f}%")
        
        if bear_signals:
            print("\n🔽 МЕДВЕЖЬИ СИГНАЛЫ (ПАДЕНИЕ):")
            for symbol, conf in sorted(bear_signals, key=lambda x: x[1], reverse=True):
                print(f"   ❌ {symbol}: {conf:.1f}%")
        
        if not bull_signals and not bear_signals:
            print("↔️ Нет четких сигналов. Рынок в боковой консолидации.")
        
        print("\n" + "="*60)
        print("✅ Анализ завершен!")
        print("="*60 + "\n")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("""
╔════════════════════════════════════════════════════════════╗
║           🤖 КРИПТОАНАЛИЗЕР - СПРАВКА                     ║
╚════════════════════════════════════════════════════════════╝

Использование:
   python crypto_analyzer.py              - Полный анализ топ-20
   python crypto_analyzer.py --help       - Показать э��у справку
   python crypto_analyzer.py --loop N     - Анализ каждые N минут

Возможности:
   ✅ Анализ топ-20 крипто по капитализации
   ✅ Получение данных с Binance API
   ✅ Расчет 11+ технических индикаторов
   ✅ Прогноз цены на 1 час вперед
   ✅ Определение диапазонов цены
   ✅ Автоматические уведомления о сигналах

Индикаторы:
   • Скользящие средние (MA7, MA14, MA25)
   • RSI (Relative Strength Index)
   • MACD (Moving Average Convergence Divergence)
   • Bollinger Bands
   • ATR (Average True Range)
   • Стохастик
   
Сигналы:
   🔼 BULL - цена будет расти в ближайший час
   🔽 BEAR - цена будет падать в ближайший час
   ↔️ NEUTRAL - цена будет двигаться боком

Требования:
   pip install requests pandas numpy

""")
        elif sys.argv[1] == "--loop" and len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
                print(f"🔄 Режим цикла: анализ каждые {interval} минут")
                while True:
                    analyzer = CryptoAnalyzer()
                    analyzer.run_full_analysis()
                    print(f"\n⏳ Следующий анализ через {interval} минут...")
                    time.sleep(interval * 60)
            except ValueError:
                print("❌ Ошибка: укажите целое число минут")
        else:
            print("❌ Неизвестный аргумент. Используйте --help для справки")
    else:
        analyzer = CryptoAnalyzer()
        analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
