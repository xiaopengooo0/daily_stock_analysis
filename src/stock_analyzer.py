# -*- coding: utf-8 -*-
"""
===================================
趋势交易分析器 - 基于用户交易理念
===================================

交易理念核心原则：
1. 严进策略 - 不追高，追求每笔交易成功率
2. 趋势交易 - MA5>MA10>MA20 多头排列，顺势而为
3. 效率优先 - 关注筹码结构好的股票
4. 买点偏好 - 在 MA5/MA10 附近回踩买入

技术标准：
- 多头排列：MA5 > MA10 > MA20
- 乖离率：(Close - MA5) / MA5 < 5%（不追高）
- 量能形态：缩量回调优先
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

import pandas as pd
import numpy as np

from src.config import get_config

logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    """趋势状态枚举"""
    STRONG_BULL = "强势多头"      # MA5 > MA10 > MA20，且间距扩大
    BULL = "多头排列"             # MA5 > MA10 > MA20
    WEAK_BULL = "弱势多头"        # MA5 > MA10，但 MA10 < MA20
    CONSOLIDATION = "盘整"        # 均线缠绕
    WEAK_BEAR = "弱势空头"        # MA5 < MA10，但 MA10 > MA20
    BEAR = "空头排列"             # MA5 < MA10 < MA20
    STRONG_BEAR = "强势空头"      # MA5 < MA10 < MA20，且间距扩大


class VolumeStatus(Enum):
    """量能状态枚举"""
    HEAVY_VOLUME_UP = "放量上涨"       # 量价齐升
    HEAVY_VOLUME_DOWN = "放量下跌"     # 放量杀跌
    SHRINK_VOLUME_UP = "缩量上涨"      # 无量上涨
    SHRINK_VOLUME_DOWN = "缩量回调"    # 缩量回调（好）
    NORMAL = "量能正常"


class BuySignal(Enum):
    """买入信号枚举"""
    STRONG_BUY = "强烈买入"       # 多条件满足
    BUY = "买入"                  # 基本条件满足
    HOLD = "持有"                 # 已持有可继续
    WAIT = "观望"                 # 等待更好时机
    SELL = "卖出"                 # 趋势转弱
    STRONG_SELL = "强烈卖出"      # 趋势破坏


class MACDStatus(Enum):
    """MACD状态枚举"""
    GOLDEN_CROSS_ZERO = "零轴上金叉"      # DIF上穿DEA，且在零轴上方
    GOLDEN_CROSS = "金叉"                # DIF上穿DEA
    BULLISH = "多头"                    # DIF>DEA>0
    CROSSING_UP = "上穿零轴"             # DIF上穿零轴
    CROSSING_DOWN = "下穿零轴"           # DIF下穿零轴
    BEARISH = "空头"                    # DIF<DEA<0
    DEATH_CROSS = "死叉"                # DIF下穿DEA


class RSIStatus(Enum):
    """RSI状态枚举"""
    OVERBOUGHT = "超买"        # RSI > 70
    STRONG_BUY = "强势买入"    # 50 < RSI < 70
    NEUTRAL = "中性"          # 40 <= RSI <= 60
    WEAK = "弱势"             # 30 < RSI < 40
    OVERSOLD = "超卖"         # RSI < 30


@dataclass
class TrendAnalysisResult:
    """趋势分析结果"""
    code: str
    
    # 趋势判断
    trend_status: TrendStatus = TrendStatus.CONSOLIDATION
    ma_alignment: str = ""           # 均线排列描述
    trend_strength: float = 0.0      # 趋势强度 0-100
    
    # 均线数据
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    current_price: float = 0.0
    
    # 乖离率（与 MA5 的偏离度）
    bias_ma5: float = 0.0            # (Close - MA5) / MA5 * 100
    bias_ma10: float = 0.0
    bias_ma20: float = 0.0
    
    # 量能分析
    volume_status: VolumeStatus = VolumeStatus.NORMAL
    volume_ratio_5d: float = 0.0     # 当日成交量/5日均量
    volume_trend: str = ""           # 量能趋势描述
    
    # 支撑压力
    support_ma5: bool = False        # MA5 是否构成支撑
    support_ma10: bool = False       # MA10 是否构成支撑
    resistance_levels: List[float] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)

    # MACD 指标
    macd_dif: float = 0.0          # DIF 快线
    macd_dea: float = 0.0          # DEA 慢线
    macd_bar: float = 0.0           # MACD 柱状图
    macd_status: MACDStatus = MACDStatus.BULLISH
    macd_signal: str = ""            # MACD 信号描述

    # RSI 指标
    rsi_6: float = 0.0              # RSI(6) 短期
    rsi_12: float = 0.0             # RSI(12) 中期
    rsi_24: float = 0.0             # RSI(24) 长期
    rsi_status: RSIStatus = RSIStatus.NEUTRAL
    rsi_signal: str = ""              # RSI 信号描述

    # K线形态识别
    kline_pattern: str = ""            # K线形态名称（如"看涨吞没"）
    kline_signal: str = ""             # K线形态信号描述

    # 布林带
    boll_upper: float = 0.0            # 布林带上轨
    boll_middle: float = 0.0           # 布林带中轨（MA20）
    boll_lower: float = 0.0            # 布林带下轨
    boll_position: str = ""            # 价格在布林带中的位置描述

    # ATR 波动率
    atr_14: float = 0.0               # ATR(14) 平均真实波幅
    atr_stop_loss_hint: float = 0.0   # 基于 ATR 的止损参考位（当前价 - 2*ATR）

    # KDJ 指标
    kdj_k: float = 0.0                # K 值
    kdj_d: float = 0.0                # D 值
    kdj_j: float = 0.0                # J 值
    kdj_signal: str = ""              # KDJ 信号描述

    # OBV 能量潮
    obv_trend: str = ""               # OBV 趋势描述（上升/下降/平缓）
    obv_price_divergence: bool = False # OBV 与价格是否背离

    # 买入信号
    buy_signal: BuySignal = BuySignal.WAIT
    signal_score: int = 0            # 综合评分 0-100
    signal_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'trend_status': self.trend_status.value,
            'ma_alignment': self.ma_alignment,
            'trend_strength': self.trend_strength,
            'ma5': self.ma5,
            'ma10': self.ma10,
            'ma20': self.ma20,
            'ma60': self.ma60,
            'current_price': self.current_price,
            'bias_ma5': self.bias_ma5,
            'bias_ma10': self.bias_ma10,
            'bias_ma20': self.bias_ma20,
            'volume_status': self.volume_status.value,
            'volume_ratio_5d': self.volume_ratio_5d,
            'volume_trend': self.volume_trend,
            'support_ma5': self.support_ma5,
            'support_ma10': self.support_ma10,
            'buy_signal': self.buy_signal.value,
            'signal_score': self.signal_score,
            'signal_reasons': self.signal_reasons,
            'risk_factors': self.risk_factors,
            'macd_dif': self.macd_dif,
            'macd_dea': self.macd_dea,
            'macd_bar': self.macd_bar,
            'macd_status': self.macd_status.value,
            'macd_signal': self.macd_signal,
            'rsi_6': self.rsi_6,
            'rsi_12': self.rsi_12,
            'rsi_24': self.rsi_24,
            'rsi_status': self.rsi_status.value,
            'rsi_signal': self.rsi_signal,
            'kline_pattern': self.kline_pattern,
            'kline_signal': self.kline_signal,
            'boll_upper': self.boll_upper,
            'boll_middle': self.boll_middle,
            'boll_lower': self.boll_lower,
            'boll_position': self.boll_position,
            'atr_14': self.atr_14,
            'atr_stop_loss_hint': self.atr_stop_loss_hint,
            'kdj_k': self.kdj_k,
            'kdj_d': self.kdj_d,
            'kdj_j': self.kdj_j,
            'kdj_signal': self.kdj_signal,
            'obv_trend': self.obv_trend,
            'obv_price_divergence': self.obv_price_divergence,
        }


class StockTrendAnalyzer:
    """
    股票趋势分析器

    基于用户交易理念实现：
    1. 趋势判断 - MA5>MA10>MA20 多头排列
    2. 乖离率检测 - 不追高，偏离 MA5 超过 5% 不买
    3. 量能分析 - 偏好缩量回调
    4. 买点识别 - 回踩 MA5/MA10 支撑
    5. MACD 指标 - 趋势确认和金叉死叉信号
    6. RSI 指标 - 超买超卖判断
    """
    
    # 交易参数配置（BIAS_THRESHOLD 从 Config 读取，见 _generate_signal）
    VOLUME_SHRINK_RATIO = 0.7   # 缩量判断阈值（当日量/5日均量）
    VOLUME_HEAVY_RATIO = 1.5    # 放量判断阈值
    MA_SUPPORT_TOLERANCE = 0.02  # MA 支撑判断容忍度（2%）

    # MACD 参数（标准12/26/9）
    MACD_FAST = 12              # 快线周期
    MACD_SLOW = 26             # 慢线周期
    MACD_SIGNAL = 9             # 信号线周期

    # RSI 参数
    RSI_SHORT = 6               # 短期RSI周期
    RSI_MID = 12               # 中期RSI周期
    RSI_LONG = 24              # 长期RSI周期
    RSI_OVERBOUGHT = 70        # 超买阈值
    RSI_OVERSOLD = 30          # 超卖阈值
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def analyze(self, df: pd.DataFrame, code: str) -> TrendAnalysisResult:
        """
        分析股票趋势
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            code: 股票代码
            
        Returns:
            TrendAnalysisResult 分析结果
        """
        result = TrendAnalysisResult(code=code)
        
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"{code} 数据不足，无法进行趋势分析")
            result.risk_factors.append("数据不足，无法完成分析")
            return result
        
        # 确保数据按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算均线
        df = self._calculate_mas(df)

        # 计算 MACD 和 RSI
        df = self._calculate_macd(df)
        df = self._calculate_rsi(df)

        # 计算扩展指标
        df = self._calculate_bollinger(df)
        df = self._calculate_atr(df)
        df = self._calculate_kdj(df)
        df = self._calculate_obv(df)

        # 获取最新数据
        latest = df.iloc[-1]
        result.current_price = float(latest['close'])
        result.ma5 = float(latest['MA5'])
        result.ma10 = float(latest['MA10'])
        result.ma20 = float(latest['MA20'])
        result.ma60 = float(latest.get('MA60', 0))

        # 1. 趋势判断
        self._analyze_trend(df, result)

        # 2. 乖离率计算
        self._calculate_bias(result)

        # 3. 量能分析
        self._analyze_volume(df, result)

        # 4. 支撑压力分析
        self._analyze_support_resistance(df, result)

        # 5. MACD 分析
        self._analyze_macd(df, result)

        # 6. RSI 分析
        self._analyze_rsi(df, result)

        # 7. K线形态分析
        self._analyze_kline_pattern(df, result)

        # 8. 布林带分析
        self._analyze_bollinger(df, result)

        # 9. ATR 波动率分析
        self._analyze_atr(df, result)

        # 10. KDJ 分析
        self._analyze_kdj(df, result)

        # 11. OBV 能量潮分析
        self._analyze_obv(df, result)

        # 12. 生成买入信号
        self._generate_signal(result)

        return result
    
    def _calculate_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算均线"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 60:
            df['MA60'] = df['close'].rolling(window=60).mean()
        else:
            df['MA60'] = df['MA20']  # 数据不足时使用 MA20 替代
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 MACD 指标

        公式：
        - EMA(12)：12日指数移动平均
        - EMA(26)：26日指数移动平均
        - DIF = EMA(12) - EMA(26)
        - DEA = EMA(DIF, 9)
        - MACD = (DIF - DEA) * 2
        """
        df = df.copy()

        # 计算快慢线 EMA
        ema_fast = df['close'].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.MACD_SLOW, adjust=False).mean()

        # 计算快线 DIF
        df['MACD_DIF'] = ema_fast - ema_slow

        # 计算信号线 DEA
        df['MACD_DEA'] = df['MACD_DIF'].ewm(span=self.MACD_SIGNAL, adjust=False).mean()

        # 计算柱状图
        df['MACD_BAR'] = (df['MACD_DIF'] - df['MACD_DEA']) * 2

        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 RSI 指标

        公式：
        - RS = 平均上涨幅度 / 平均下跌幅度
        - RSI = 100 - (100 / (1 + RS))
        """
        df = df.copy()

        for period in [self.RSI_SHORT, self.RSI_MID, self.RSI_LONG]:
            # 计算价格变化
            delta = df['close'].diff()

            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 计算平均涨跌幅
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # 计算 RS 和 RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # 填充 NaN 值
            rsi = rsi.fillna(50)  # 默认中性值

            # 添加到 DataFrame
            col_name = f'RSI_{period}'
            df[col_name] = rsi

        return df
    
    def _analyze_trend(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析趋势状态
        
        核心逻辑：判断均线排列和趋势强度
        """
        ma5, ma10, ma20 = result.ma5, result.ma10, result.ma20
        
        # 判断均线排列
        if ma5 > ma10 > ma20:
            # 检查间距是否在扩大（强势）
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA5'] - prev['MA20']) / prev['MA20'] * 100 if prev['MA20'] > 0 else 0
            curr_spread = (ma5 - ma20) / ma20 * 100 if ma20 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BULL
                result.ma_alignment = "强势多头排列，均线发散上行"
                result.trend_strength = 90
            else:
                result.trend_status = TrendStatus.BULL
                result.ma_alignment = "多头排列 MA5>MA10>MA20"
                result.trend_strength = 75
                
        elif ma5 > ma10 and ma10 <= ma20:
            result.trend_status = TrendStatus.WEAK_BULL
            result.ma_alignment = "弱势多头，MA5>MA10 但 MA10≤MA20"
            result.trend_strength = 55
            
        elif ma5 < ma10 < ma20:
            prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]
            prev_spread = (prev['MA20'] - prev['MA5']) / prev['MA5'] * 100 if prev['MA5'] > 0 else 0
            curr_spread = (ma20 - ma5) / ma5 * 100 if ma5 > 0 else 0
            
            if curr_spread > prev_spread and curr_spread > 5:
                result.trend_status = TrendStatus.STRONG_BEAR
                result.ma_alignment = "强势空头排列，均线发散下行"
                result.trend_strength = 10
            else:
                result.trend_status = TrendStatus.BEAR
                result.ma_alignment = "空头排列 MA5<MA10<MA20"
                result.trend_strength = 25
                
        elif ma5 < ma10 and ma10 >= ma20:
            result.trend_status = TrendStatus.WEAK_BEAR
            result.ma_alignment = "弱势空头，MA5<MA10 但 MA10≥MA20"
            result.trend_strength = 40
            
        else:
            result.trend_status = TrendStatus.CONSOLIDATION
            result.ma_alignment = "均线缠绕，趋势不明"
            result.trend_strength = 50
    
    def _calculate_bias(self, result: TrendAnalysisResult) -> None:
        """
        计算乖离率
        
        乖离率 = (现价 - 均线) / 均线 * 100%
        
        严进策略：乖离率超过 5% 不追高
        """
        price = result.current_price
        
        if result.ma5 > 0:
            result.bias_ma5 = (price - result.ma5) / result.ma5 * 100
        if result.ma10 > 0:
            result.bias_ma10 = (price - result.ma10) / result.ma10 * 100
        if result.ma20 > 0:
            result.bias_ma20 = (price - result.ma20) / result.ma20 * 100
    
    def _analyze_volume(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析量能
        
        偏好：缩量回调 > 放量上涨 > 缩量上涨 > 放量下跌
        """
        if len(df) < 5:
            return
        
        latest = df.iloc[-1]
        vol_5d_avg = df['volume'].iloc[-6:-1].mean()
        
        if vol_5d_avg > 0:
            result.volume_ratio_5d = float(latest['volume']) / vol_5d_avg
        
        # 判断价格变化
        prev_close = df.iloc[-2]['close']
        price_change = (latest['close'] - prev_close) / prev_close * 100
        
        # 量能状态判断
        if result.volume_ratio_5d >= self.VOLUME_HEAVY_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_UP
                result.volume_trend = "放量上涨，多头力量强劲"
            else:
                result.volume_status = VolumeStatus.HEAVY_VOLUME_DOWN
                result.volume_trend = "放量下跌，注意风险"
        elif result.volume_ratio_5d <= self.VOLUME_SHRINK_RATIO:
            if price_change > 0:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_UP
                result.volume_trend = "缩量上涨，上攻动能不足"
            else:
                result.volume_status = VolumeStatus.SHRINK_VOLUME_DOWN
                result.volume_trend = "缩量回调，洗盘特征明显（好）"
        else:
            result.volume_status = VolumeStatus.NORMAL
            result.volume_trend = "量能正常"
    
    def _analyze_support_resistance(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析支撑压力位
        
        买点偏好：回踩 MA5/MA10 获得支撑
        """
        price = result.current_price
        
        # 检查是否在 MA5 附近获得支撑
        if result.ma5 > 0:
            ma5_distance = abs(price - result.ma5) / result.ma5
            if ma5_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma5:
                result.support_ma5 = True
                result.support_levels.append(result.ma5)
        
        # 检查是否在 MA10 附近获得支撑
        if result.ma10 > 0:
            ma10_distance = abs(price - result.ma10) / result.ma10
            if ma10_distance <= self.MA_SUPPORT_TOLERANCE and price >= result.ma10:
                result.support_ma10 = True
                if result.ma10 not in result.support_levels:
                    result.support_levels.append(result.ma10)
        
        # MA20 作为重要支撑
        if result.ma20 > 0 and price >= result.ma20:
            result.support_levels.append(result.ma20)
        
        # 近期高点作为压力
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            if recent_high > price:
                result.resistance_levels.append(recent_high)

    def _analyze_macd(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析 MACD 指标

        核心信号：
        - 零轴上金叉：最强买入信号
        - 金叉：DIF 上穿 DEA
        - 死叉：DIF 下穿 DEA
        """
        if len(df) < self.MACD_SLOW:
            result.macd_signal = "数据不足"
            return

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 获取 MACD 数据
        result.macd_dif = float(latest['MACD_DIF'])
        result.macd_dea = float(latest['MACD_DEA'])
        result.macd_bar = float(latest['MACD_BAR'])

        # 判断金叉死叉
        prev_dif_dea = prev['MACD_DIF'] - prev['MACD_DEA']
        curr_dif_dea = result.macd_dif - result.macd_dea

        # 金叉：DIF 上穿 DEA
        is_golden_cross = prev_dif_dea <= 0 and curr_dif_dea > 0

        # 死叉：DIF 下穿 DEA
        is_death_cross = prev_dif_dea >= 0 and curr_dif_dea < 0

        # 零轴穿越
        prev_zero = prev['MACD_DIF']
        curr_zero = result.macd_dif
        is_crossing_up = prev_zero <= 0 and curr_zero > 0
        is_crossing_down = prev_zero >= 0 and curr_zero < 0

        # 判断 MACD 状态
        if is_golden_cross and curr_zero > 0:
            result.macd_status = MACDStatus.GOLDEN_CROSS_ZERO
            result.macd_signal = "⭐ 零轴上金叉，强烈买入信号！"
        elif is_crossing_up:
            result.macd_status = MACDStatus.CROSSING_UP
            result.macd_signal = "⚡ DIF上穿零轴，趋势转强"
        elif is_golden_cross:
            result.macd_status = MACDStatus.GOLDEN_CROSS
            result.macd_signal = "✅ 金叉，趋势向上"
        elif is_death_cross:
            result.macd_status = MACDStatus.DEATH_CROSS
            result.macd_signal = "❌ 死叉，趋势向下"
        elif is_crossing_down:
            result.macd_status = MACDStatus.CROSSING_DOWN
            result.macd_signal = "⚠️ DIF下穿零轴，趋势转弱"
        elif result.macd_dif > 0 and result.macd_dea > 0:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = "✓ 多头排列，持续上涨"
        elif result.macd_dif < 0 and result.macd_dea < 0:
            result.macd_status = MACDStatus.BEARISH
            result.macd_signal = "⚠ 空头排列，持续下跌"
        else:
            result.macd_status = MACDStatus.BULLISH
            result.macd_signal = " MACD 中性区域"

    def _analyze_rsi(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        分析 RSI 指标

        核心判断：
        - RSI > 70：超买，谨慎追高
        - RSI < 30：超卖，关注反弹
        - 40-60：中性区域
        """
        if len(df) < self.RSI_LONG:
            result.rsi_signal = "数据不足"
            return

        latest = df.iloc[-1]

        # 获取 RSI 数据
        result.rsi_6 = float(latest[f'RSI_{self.RSI_SHORT}'])
        result.rsi_12 = float(latest[f'RSI_{self.RSI_MID}'])
        result.rsi_24 = float(latest[f'RSI_{self.RSI_LONG}'])

        # 以中期 RSI(12) 为主进行判断
        rsi_mid = result.rsi_12

        # 判断 RSI 状态
        if rsi_mid > self.RSI_OVERBOUGHT:
            result.rsi_status = RSIStatus.OVERBOUGHT
            result.rsi_signal = f"⚠️ RSI超买({rsi_mid:.1f}>70)，短期回调风险高"
        elif rsi_mid > 60:
            result.rsi_status = RSIStatus.STRONG_BUY
            result.rsi_signal = f"✅ RSI强势({rsi_mid:.1f})，多头力量充足"
        elif rsi_mid >= 40:
            result.rsi_status = RSIStatus.NEUTRAL
            result.rsi_signal = f" RSI中性({rsi_mid:.1f})，震荡整理中"
        elif rsi_mid >= self.RSI_OVERSOLD:
            result.rsi_status = RSIStatus.WEAK
            result.rsi_signal = f"⚡ RSI弱势({rsi_mid:.1f})，关注反弹"
        else:
            result.rsi_status = RSIStatus.OVERSOLD
            result.rsi_signal = f"⭐ RSI超卖({rsi_mid:.1f}<30)，反弹机会大"

    def _generate_signal(self, result: TrendAnalysisResult) -> None:
        """
        生成买入信号

        综合评分系统：
        - 趋势（30分）：多头排列得分高
        - 乖离率（20分）：接近 MA5 得分高
        - 量能（15分）：缩量回调得分高
        - 支撑（10分）：获得均线支撑得分高
        - MACD（15分）：金叉和多头得分高
        - RSI（10分）：超卖和强势得分高
        """
        score = 0
        reasons = []
        risks = []

        # === 趋势评分（30分）===
        trend_scores = {
            TrendStatus.STRONG_BULL: 30,
            TrendStatus.BULL: 26,
            TrendStatus.WEAK_BULL: 18,
            TrendStatus.CONSOLIDATION: 12,
            TrendStatus.WEAK_BEAR: 8,
            TrendStatus.BEAR: 4,
            TrendStatus.STRONG_BEAR: 0,
        }
        trend_score = trend_scores.get(result.trend_status, 12)
        score += trend_score

        if result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            reasons.append(f"✅ {result.trend_status.value}，顺势做多")
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            risks.append(f"⚠️ {result.trend_status.value}，不宜做多")

        # === 乖离率评分（20分，强势趋势补偿）===
        bias = result.bias_ma5
        if bias != bias or bias is None:  # NaN or None defense
            bias = 0.0
        base_threshold = get_config().bias_threshold

        # Strong trend compensation: relax threshold for STRONG_BULL with high strength
        trend_strength = result.trend_strength if result.trend_strength == result.trend_strength else 0.0
        if result.trend_status == TrendStatus.STRONG_BULL and (trend_strength or 0) >= 70:
            effective_threshold = base_threshold * 1.5
            is_strong_trend = True
        else:
            effective_threshold = base_threshold
            is_strong_trend = False

        if bias < 0:
            # Price below MA5 (pullback)
            if bias > -3:
                score += 20
                reasons.append(f"✅ 价格略低于MA5({bias:.1f}%)，回踩买点")
            elif bias > -5:
                score += 16
                reasons.append(f"✅ 价格回踩MA5({bias:.1f}%)，观察支撑")
            else:
                score += 8
                risks.append(f"⚠️ 乖离率过大({bias:.1f}%)，可能破位")
        elif bias < 2:
            score += 18
            reasons.append(f"✅ 价格贴近MA5({bias:.1f}%)，介入好时机")
        elif bias < base_threshold:
            score += 14
            reasons.append(f"⚡ 价格略高于MA5({bias:.1f}%)，可小仓介入")
        elif bias > effective_threshold:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{effective_threshold:.1f}%)，严禁追高！"
            )
        elif bias > base_threshold and is_strong_trend:
            score += 10
            reasons.append(
                f"⚡ 强势趋势中乖离率偏高({bias:.1f}%)，可轻仓追踪"
            )
        else:
            score += 4
            risks.append(
                f"❌ 乖离率过高({bias:.1f}%>{base_threshold:.1f}%)，严禁追高！"
            )

        # === 量能评分（15分）===
        volume_scores = {
            VolumeStatus.SHRINK_VOLUME_DOWN: 15,  # 缩量回调最佳
            VolumeStatus.HEAVY_VOLUME_UP: 12,     # 放量上涨次之
            VolumeStatus.NORMAL: 10,
            VolumeStatus.SHRINK_VOLUME_UP: 6,     # 无量上涨较差
            VolumeStatus.HEAVY_VOLUME_DOWN: 0,    # 放量下跌最差
        }
        vol_score = volume_scores.get(result.volume_status, 8)
        score += vol_score

        if result.volume_status == VolumeStatus.SHRINK_VOLUME_DOWN:
            reasons.append("✅ 缩量回调，主力洗盘")
        elif result.volume_status == VolumeStatus.HEAVY_VOLUME_DOWN:
            risks.append("⚠️ 放量下跌，注意风险")

        # === 支撑评分（10分）===
        if result.support_ma5:
            score += 5
            reasons.append("✅ MA5支撑有效")
        if result.support_ma10:
            score += 5
            reasons.append("✅ MA10支撑有效")

        # === MACD 评分（15分）===
        macd_scores = {
            MACDStatus.GOLDEN_CROSS_ZERO: 15,  # 零轴上金叉最强
            MACDStatus.GOLDEN_CROSS: 12,      # 金叉
            MACDStatus.CROSSING_UP: 10,       # 上穿零轴
            MACDStatus.BULLISH: 8,            # 多头
            MACDStatus.BEARISH: 2,            # 空头
            MACDStatus.CROSSING_DOWN: 0,       # 下穿零轴
            MACDStatus.DEATH_CROSS: 0,        # 死叉
        }
        macd_score = macd_scores.get(result.macd_status, 5)
        score += macd_score

        if result.macd_status in [MACDStatus.GOLDEN_CROSS_ZERO, MACDStatus.GOLDEN_CROSS]:
            reasons.append(f"✅ {result.macd_signal}")
        elif result.macd_status in [MACDStatus.DEATH_CROSS, MACDStatus.CROSSING_DOWN]:
            risks.append(f"⚠️ {result.macd_signal}")
        else:
            reasons.append(result.macd_signal)

        # === RSI 评分（10分）===
        rsi_scores = {
            RSIStatus.OVERSOLD: 10,       # 超卖最佳
            RSIStatus.STRONG_BUY: 8,     # 强势
            RSIStatus.NEUTRAL: 5,        # 中性
            RSIStatus.WEAK: 3,            # 弱势
            RSIStatus.OVERBOUGHT: 0,       # 超买最差
        }
        rsi_score = rsi_scores.get(result.rsi_status, 5)
        score += rsi_score

        if result.rsi_status in [RSIStatus.OVERSOLD, RSIStatus.STRONG_BUY]:
            reasons.append(f"✅ {result.rsi_signal}")
        elif result.rsi_status == RSIStatus.OVERBOUGHT:
            risks.append(f"⚠️ {result.rsi_signal}")
        else:
            reasons.append(result.rsi_signal)

        # === 综合判断 ===
        result.signal_score = score
        result.signal_reasons = reasons
        result.risk_factors = risks

        # 生成买入信号（调整阈值以适应新的100分制）
        if score >= 75 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL]:
            result.buy_signal = BuySignal.STRONG_BUY
        elif score >= 60 and result.trend_status in [TrendStatus.STRONG_BULL, TrendStatus.BULL, TrendStatus.WEAK_BULL]:
            result.buy_signal = BuySignal.BUY
        elif score >= 45:
            result.buy_signal = BuySignal.HOLD
        elif score >= 30:
            result.buy_signal = BuySignal.WAIT
        elif result.trend_status in [TrendStatus.BEAR, TrendStatus.STRONG_BEAR]:
            result.buy_signal = BuySignal.STRONG_SELL
        else:
            result.buy_signal = BuySignal.SELL

    # ========================================
    # 扩展指标计算方法
    # ========================================

    def _calculate_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带（MA20 ± 2σ）"""
        if df is None or len(df) < 20:
            return df
        df = df.copy()
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['BOLL_UPPER'] = ma20 + 2 * std20
        df['BOLL_MIDDLE'] = ma20
        df['BOLL_LOWER'] = ma20 - 2 * std20
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 ATR（平均真实波幅）"""
        if df is None or len(df) < period + 1:
            return df
        df = df.copy()
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=period).mean()
        return df

    def _calculate_kdj(self, df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
        """计算 KDJ 指标"""
        if df is None or len(df) < period:
            return df
        df = df.copy()
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)

        k_values = [50.0]
        d_values = [50.0]
        for i in range(1, len(df)):
            k = 2 / 3 * k_values[-1] + 1 / 3 * rsv.iloc[i]
            d = 2 / 3 * d_values[-1] + 1 / 3 * k
            k_values.append(k)
            d_values.append(d)

        df['KDJ_K'] = k_values
        df['KDJ_D'] = d_values
        df['KDJ_J'] = 3 * pd.Series(k_values) - 2 * pd.Series(d_values)
        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 OBV（能量潮）"""
        if df is None or len(df) < 2:
            return df
        df = df.copy()
        obv = [0.0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + float(df['volume'].iloc[i]))
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - float(df['volume'].iloc[i]))
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df

    # ========================================
    # 扩展指标分析方法
    # ========================================

    def _analyze_kline_pattern(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """
        识别 K 线形态（吞没、十字星、锤子线、早晨之星、黄昏之星等）
        """
        if df is None or len(df) < 3:
            return

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        c_open = float(curr['open'])
        c_close = float(curr['close'])
        c_high = float(curr['high'])
        c_low = float(curr['low'])
        p_open = float(prev['open'])
        p_close = float(prev['close'])

        c_body = abs(c_close - c_open)
        c_range = c_high - c_low if c_high != c_low else 0.001
        p_body = abs(p_close - p_open)

        detected = []

        # 十字星（Doji）
        if c_body < c_range * 0.05 and c_range > 0:
            detected.append(("十字星", "⭐ 十字星形态，多空分歧，趋势可能反转", 1))

        # 看涨吞没（Bullish Engulfing）
        if p_close < p_open and c_close > c_open:
            if c_close >= p_open and c_open <= p_close and c_body > p_body:
                detected.append(("看涨吞没", "⭐ 看涨吞没形态，多头强势反转信号", 3))

        # 看跌吞没（Bearish Engulfing）
        if p_close > p_open and c_close < c_open:
            if c_close <= p_open and c_open >= p_close and c_body > p_body:
                detected.append(("看跌吞没", "⚠️ 看跌吞没形态，空头强势反转信号", 3))

        # 锤子线（Hammer）
        upper_shadow = c_high - max(c_open, c_close)
        lower_shadow = min(c_open, c_close) - c_low
        if c_body > 0 and lower_shadow >= 2 * c_body and upper_shadow < c_body * 0.5:
            detected.append(("锤子线", "⭐ 锤子线形态，下跌中可能出现反转", 2))

        # 倒锤子（Inverted Hammer）
        if c_body > 0 and upper_shadow >= 2 * c_body and lower_shadow < c_body * 0.5:
            detected.append(("倒锤子", "⚡ 倒锤子形态，上涨中需警惕回调", 2))

        # 射击之星（Shooting Star）— 出现在上涨趋势中
        if c_body > 0 and upper_shadow >= 2 * c_body and lower_shadow < c_body * 0.3:
            detected.append(("射击之星", "⚠️ 射击之星形态，上涨动能衰竭", 2))

        # 早晨之星（Morning Star）— 3 根 K 线
        if len(df) >= 3:
            bar0 = df.iloc[-3]
            bar1 = df.iloc[-2]
            bar2 = df.iloc[-1]
            b0_body = abs(float(bar0['close']) - float(bar0['open']))
            b0_bearish = float(bar0['close']) < float(bar0['open'])
            b1_body = abs(float(bar1['close']) - float(bar1['open']))
            b2_bullish = float(bar2['close']) > float(bar2['open'])
            b2_body = abs(float(bar2['close']) - float(bar2['open']))
            if b0_bearish and b0_body > 0 and b1_body < b0_body * 0.3 and b2_bullish and b2_body > b0_body * 0.5:
                detected.append(("早晨之星", "⭐ 早晨之星形态，强烈看涨反转信号", 4))

            # 黄昏之星（Evening Star）
            b0_bullish = float(bar0['close']) > float(bar0['open'])
            b2_bearish = float(bar2['close']) < float(bar2['open'])
            if b0_bullish and b0_body > 0 and b1_body < b0_body * 0.3 and b2_bearish and b2_body > b0_body * 0.5:
                detected.append(("黄昏之星", "⚠️ 黄昏之星形态，强烈看跌反转信号", 4))

        if detected:
            best = max(detected, key=lambda x: x[2])
            result.kline_pattern = best[0]
            result.kline_signal = best[1]

    def _analyze_bollinger(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """分析布林带位置"""
        if df is None or 'BOLL_UPPER' not in df.columns:
            return
        latest = df.iloc[-1]
        upper = float(latest.get('BOLL_UPPER', 0))
        middle = float(latest.get('BOLL_MIDDLE', 0))
        lower = float(latest.get('BOLL_LOWER', 0))
        price = float(latest['close'])

        # NaN 防护：数据不足时 pandas 计算结果为 NaN
        if math.isnan(upper) or math.isnan(lower) or math.isnan(price):
            return

        if upper == 0 or lower == 0:
            return

        result.boll_upper = upper
        result.boll_middle = middle
        result.boll_lower = lower

        bandwidth = upper - lower
        if bandwidth <= 0:
            return

        if price > upper:
            result.boll_position = f"突破上轨（{price:.2f} > {upper:.2f}），短期超买"
        elif price > middle + (upper - middle) * 0.7:
            result.boll_position = f"接近上轨，偏强运行"
        elif price >= middle:
            result.boll_position = f"中轨上方运行，偏多"
        elif price >= lower + (middle - lower) * 0.3:
            result.boll_position = f"中轨下方运行，偏弱"
        elif price > lower:
            result.boll_position = f"接近下轨，偏弱运行"
        else:
            result.boll_position = f"跌破下轨（{price:.2f} < {lower:.2f}），短期超卖"

    def _analyze_atr(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """分析 ATR 波动率"""
        if df is None or 'ATR' not in df.columns:
            return
        latest = df.iloc[-1]
        atr = float(latest.get('ATR', 0))
        # NaN 防护
        if math.isnan(atr) or atr <= 0:
            return
        result.atr_14 = atr
        price = float(latest['close'])
        if not math.isnan(price) and price > 0:
            result.atr_stop_loss_hint = round(price - 2 * atr, 2)

    def _analyze_kdj(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """分析 KDJ 指标"""
        if df is None or 'KDJ_K' not in df.columns:
            return
        latest = df.iloc[-1]
        k_val = float(latest.get('KDJ_K', 50))
        d_val = float(latest.get('KDJ_D', 50))
        j_val = float(latest.get('KDJ_J', 50))

        # NaN 防护
        if math.isnan(k_val) or math.isnan(d_val) or math.isnan(j_val):
            return

        result.kdj_k = k_val
        result.kdj_d = d_val
        result.kdj_j = j_val

        k = result.kdj_k
        d = result.kdj_d

        if len(df) >= 2:
            prev_k = float(df.iloc[-2].get('KDJ_K', 50))
            prev_d = float(df.iloc[-2].get('KDJ_D', 50))
            golden_cross = prev_k <= prev_d and k > d
            death_cross = prev_k >= prev_d and k < d

            if golden_cross and d < 20:
                result.kdj_signal = "⭐ KDJ 低位金叉，强烈买入信号"
            elif golden_cross:
                result.kdj_signal = "✅ KDJ 金叉，趋势向上"
            elif death_cross and d > 80:
                result.kdj_signal = "⚠️ KDJ 高位死叉，强烈卖出信号"
            elif death_cross:
                result.kdj_signal = "⚠️ KDJ 死叉，趋势转弱"

        if not result.kdj_signal:
            if k > 80 and d > 80:
                result.kdj_signal = "⚠️ KDJ 超买区域"
            elif k < 20 and d < 20:
                result.kdj_signal = "⚡ KDJ 超卖区域"
            else:
                result.kdj_signal = f"KDJ 中性区域 (K={k:.1f})"

    def _analyze_obv(self, df: pd.DataFrame, result: TrendAnalysisResult) -> None:
        """分析 OBV 能量潮"""
        if df is None or 'OBV' not in df.columns or len(df) < 10:
            return

        obv = df['OBV']
        close = df['close']

        obv_recent = obv.iloc[-5:].values
        obv_older = obv.iloc[-10:-5].values
        close_recent = close.iloc[-5:].values
        close_older = close.iloc[-10:-5].values

        obv_recent_mean = float(np.mean(obv_recent))
        obv_older_mean = float(np.mean(obv_older))
        close_recent_mean = float(np.mean(close_recent))
        close_older_mean = float(np.mean(close_older))

        obv_rising = obv_recent_mean > obv_older_mean
        price_rising = close_recent_mean > close_older_mean

        if obv_rising and price_rising:
            result.obv_trend = "上升（量价齐升）"
        elif not obv_rising and not price_rising:
            result.obv_trend = "下降（量价齐跌）"
        elif obv_rising and not price_rising:
            result.obv_trend = "上升（缩量下跌，资金未出逃）"
        else:
            result.obv_trend = "下降（放量下跌，资金出逃）"

        result.obv_price_divergence = obv_rising != price_rising

    def format_analysis(self, result: TrendAnalysisResult) -> str:
        """
        格式化分析结果为文本

        Args:
            result: 分析结果

        Returns:
            格式化的分析文本
        """
        lines = [
            f"=== {result.code} 趋势分析 ===",
            f"",
            f"📊 趋势判断: {result.trend_status.value}",
            f"   均线排列: {result.ma_alignment}",
            f"   趋势强度: {result.trend_strength}/100",
            f"",
            f"📈 均线数据:",
            f"   现价: {result.current_price:.2f}",
            f"   MA5:  {result.ma5:.2f} (乖离 {result.bias_ma5:+.2f}%)",
            f"   MA10: {result.ma10:.2f} (乖离 {result.bias_ma10:+.2f}%)",
            f"   MA20: {result.ma20:.2f} (乖离 {result.bias_ma20:+.2f}%)",
            f"",
            f"📊 量能分析: {result.volume_status.value}",
            f"   量比(vs5日): {result.volume_ratio_5d:.2f}",
            f"   量能趋势: {result.volume_trend}",
            f"",
            f"📈 MACD指标: {result.macd_status.value}",
            f"   DIF: {result.macd_dif:.4f}",
            f"   DEA: {result.macd_dea:.4f}",
            f"   MACD: {result.macd_bar:.4f}",
            f"   信号: {result.macd_signal}",
            f"",
            f"📊 RSI指标: {result.rsi_status.value}",
            f"   RSI(6): {result.rsi_6:.1f}",
            f"   RSI(12): {result.rsi_12:.1f}",
            f"   RSI(24): {result.rsi_24:.1f}",
            f"   信号: {result.rsi_signal}",
            f"",
        ]

        if result.kline_pattern:
            lines.append(f"📈 K线形态: {result.kline_pattern}")
            lines.append(f"   {result.kline_signal}")
            lines.append(f"")

        if result.boll_position:
            lines.extend([
                f"📊 布林带: 上轨 {result.boll_upper:.2f} / 中轨 {result.boll_middle:.2f} / 下轨 {result.boll_lower:.2f}",
                f"   位置: {result.boll_position}",
                f"",
            ])

        if result.atr_14 > 0:
            lines.extend([
                f"📏 ATR(14): {result.atr_14:.2f}  ATR止损参考: {result.atr_stop_loss_hint:.2f}",
                f"",
            ])

        if result.kdj_signal:
            lines.extend([
                f"📊 KDJ: K={result.kdj_k:.1f} D={result.kdj_d:.1f} J={result.kdj_j:.1f}",
                f"   {result.kdj_signal}",
                f"",
            ])

        if result.obv_trend:
            div_text = " ⚠️ 与价格背离" if result.obv_price_divergence else ""
            lines.extend([
                f"📊 OBV: {result.obv_trend}{div_text}",
                f"",
            ])

        lines.extend([
            f"🎯 操作建议: {result.buy_signal.value}",
            f"   综合评分: {result.signal_score}/100",
        ])

        if result.signal_reasons:
            lines.append(f"")
            lines.append(f"✅ 买入理由:")
            for reason in result.signal_reasons:
                lines.append(f"   {reason}")

        if result.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ 风险因素:")
            for risk in result.risk_factors:
                lines.append(f"   {risk}")

        return "\n".join(lines)


def analyze_stock(df: pd.DataFrame, code: str) -> TrendAnalysisResult:
    """
    便捷函数：分析单只股票
    
    Args:
        df: 包含 OHLCV 数据的 DataFrame
        code: 股票代码
        
    Returns:
        TrendAnalysisResult 分析结果
    """
    analyzer = StockTrendAnalyzer()
    return analyzer.analyze(df, code)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 模拟数据测试
    import numpy as np
    
    dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
    np.random.seed(42)
    
    # 模拟多头排列的数据
    base_price = 10.0
    prices = [base_price]
    for i in range(59):
        change = np.random.randn() * 0.02 + 0.003  # 轻微上涨趋势
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(1000000, 5000000) for _ in prices],
    })
    
    analyzer = StockTrendAnalyzer()
    result = analyzer.analyze(df, '000001')
    print(analyzer.format_analysis(result))
