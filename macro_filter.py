""" 
Macro Economic Filter Module - Federal Reserve Data Integration
================================================================

Fetches key Federal Reserve data from FRED (Federal Reserve Economic Data)
and uses it as a macro-level filter for trading decisions.

Key indicators tracked:
- WALCL: Fed Total Assets (Balance Sheet) - liquidity proxy
- RRPONTSYD: Reverse Repo (ON RRP) - liquidity drain
- WTREGEN: Treasury General Account (TGA) - fiscal liquidity impact
- DFF: Federal Funds Rate - monetary policy stance
- T10Y2Y: 10Y-2Y Spread - recession indicator

Trading logic:
- Rising Fed balance + falling RRP = MORE liquidity = risk-on (favor longs)
- Falling Fed balance + rising RRP = LESS liquidity = risk-off (favor shorts/caution)
- Inverted yield curve (T10Y2Y < 0) = recession warning = reduce risk
"""

import logging
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

# FRED API - Free, no key needed for basic access via JSON endpoint
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Key series IDs
SERIES = {
    'fed_balance': 'WALCL',       # Fed Total Assets (weekly, Wed)
    'reverse_repo': 'RRPONTSYD',  # Overnight Reverse Repo (daily)
    'tga': 'WTREGEN',             # Treasury General Account (weekly, Wed)
    'fed_funds': 'DFF',           # Federal Funds Rate (daily)
    'yield_spread': 'T10Y2Y',    # 10Y-2Y Treasury Spread (daily)
}


class MacroFilter:
    """
    Fetches Federal Reserve macro data and provides a liquidity/risk assessment
    that can be used as a filter or strength modifier for trading signals.
    """

    def __init__(self, fred_api_key: str = ""):
        """
        Initialize MacroFilter.
        
        Args:
            fred_api_key: FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
                         If empty, uses limited public JSON endpoint.
        """
        self.api_key = fred_api_key
        self._cache: Dict[str, Dict] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(hours=6)  # Refresh every 6 hours
        logger.info("MacroFilter initialized")

    def _fetch_fred_series(self, series_id: str, count: int = 10) -> Optional[List[Dict]]:
        """
        Fetch recent observations from FRED.
        
        Args:
            series_id: FRED series identifier
            count: Number of recent observations
            
        Returns:
            List of {'date': str, 'value': float} dicts, or None on error
        """
        # Check cache
        now = datetime.utcnow()
        if series_id in self._cache and series_id in self._cache_expiry:
            if now < self._cache_expiry[series_id]:
                return self._cache[series_id]

        try:
            if self.api_key:
                # Official FRED API
                params = {
                    'series_id': series_id,
                    'api_key': self.api_key,
                    'file_type': 'json',
                    'sort_order': 'desc',
                    'limit': count,
                }
                resp = requests.get(FRED_BASE_URL, params=params, timeout=15)
            else:
                # Public JSON fallback (limited but works without key)
                end_date = now.strftime('%Y-%m-%d')
                start_date = (now - timedelta(days=90)).strftime('%Y-%m-%d')
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}&coed={end_date}"
                resp = requests.get(url, timeout=15)
                
                if resp.status_code == 200:
                    # Parse CSV
                    lines = resp.text.strip().split('\n')
                    if len(lines) > 1:
                        observations = []
                        for line in lines[1:]:  # Skip header
                            parts = line.split(',')
                            if len(parts) == 2 and parts[1] != '.':
                                try:
                                    observations.append({
                                        'date': parts[0],
                                        'value': float(parts[1])
                                    })
                                except ValueError:
                                    continue
                        observations.sort(key=lambda x: x['date'], reverse=True)
                        self._cache[series_id] = observations[:count]
                        self._cache_expiry[series_id] = now + self._cache_duration
                        return self._cache[series_id]
                
                logger.warning(f"FRED fetch failed for {series_id}: {resp.status_code}")
                return None

            if resp.status_code == 200:
                data = resp.json()
                observations = []
                for obs in data.get('observations', []):
                    if obs.get('value', '.') != '.':
                        observations.append({
                            'date': obs['date'],
                            'value': float(obs['value'])
                        })
                self._cache[series_id] = observations[:count]
                self._cache_expiry[series_id] = now + self._cache_duration
                return self._cache[series_id]
            else:
                logger.warning(f"FRED API error for {series_id}: {resp.status_code}")
                return None

        except Exception as e:
            logger.warning(f"FRED fetch error for {series_id}: {e}")
            return None

    def get_liquidity_assessment(self) -> Dict:
        """
        Assess current liquidity conditions based on Fed data.
        
        Net Liquidity \u2248 Fed Balance Sheet - TGA - Reverse Repo
        
        Returns:
            Dict with:
            - 'net_liquidity_trend': 'EXPANDING', 'CONTRACTING', or 'STABLE'
            - 'fed_balance_trend': 'UP', 'DOWN', or 'FLAT'
            - 'rrp_trend': 'UP', 'DOWN', or 'FLAT'
            - 'yield_curve': 'NORMAL', 'FLAT', or 'INVERTED'
            - 'risk_bias': 'RISK_ON', 'RISK_OFF', or 'NEUTRAL'
            - 'macro_score': float (-1 to +1, positive = risk-on)
            - 'details': dict with raw values
        """
        result = {
            'net_liquidity_trend': 'STABLE',
            'fed_balance_trend': 'FLAT',
            'rrp_trend': 'FLAT',
            'yield_curve': 'NORMAL',
            'risk_bias': 'NEUTRAL',
            'macro_score': 0.0,
            'details': {},
            'safe': True,  # Compatible with news_filter interface
        }

        macro_score = 0.0

        # Fed Balance Sheet trend
        fed_data = self._fetch_fred_series(SERIES['fed_balance'], 5)
        if fed_data and len(fed_data) >= 2:
            current = fed_data[0]['value']
            previous = fed_data[1]['value']
            change_pct = (current - previous) / previous * 100
            result['details']['fed_balance'] = current
            result['details']['fed_balance_change_pct'] = round(change_pct, 3)

            if change_pct > 0.1:
                result['fed_balance_trend'] = 'UP'
                macro_score += 0.3  # Expanding balance = more liquidity
            elif change_pct < -0.1:
                result['fed_balance_trend'] = 'DOWN'
                macro_score -= 0.3
            logger.info(f"Fed Balance: ${current/1e6:.1f}T ({change_pct:+.3f}%)")

        # Reverse Repo trend (high RRP = liquidity parked, not flowing)
        rrp_data = self._fetch_fred_series(SERIES['reverse_repo'], 5)
        if rrp_data and len(rrp_data) >= 2:
            current = rrp_data[0]['value']
            previous = rrp_data[-1]['value']  # Compare to oldest
            result['details']['reverse_repo'] = current

            if current < previous * 0.95:
                result['rrp_trend'] = 'DOWN'
                macro_score += 0.2  # Falling RRP = liquidity entering markets
            elif current > previous * 1.05:
                result['rrp_trend'] = 'UP'
                macro_score -= 0.2
            logger.info(f"Reverse Repo: ${current:.0f}B (trend: {result['rrp_trend']})")

        # TGA (Treasury General Account)
        tga_data = self._fetch_fred_series(SERIES['tga'], 5)
        if tga_data and len(tga_data) >= 2:
            current = tga_data[0]['value']
            previous = tga_data[1]['value']
            result['details']['tga'] = current

            if current < previous * 0.95:
                macro_score += 0.15  # Falling TGA = government spending = liquidity
            elif current > previous * 1.05:
                macro_score -= 0.15  # Rising TGA = government collecting = draining

        # Yield curve (10Y-2Y spread)
        spread_data = self._fetch_fred_series(SERIES['yield_spread'], 5)
        if spread_data and len(spread_data) >= 1:
            spread = spread_data[0]['value']
            result['details']['yield_spread'] = spread

            if spread < -0.2:
                result['yield_curve'] = 'INVERTED'
                macro_score -= 0.3  # Recession signal
            elif spread < 0.2:
                result['yield_curve'] = 'FLAT'
                macro_score -= 0.1
            else:
                result['yield_curve'] = 'NORMAL'
                macro_score += 0.1
            logger.info(f"Yield Curve: {spread:.2f}% ({result['yield_curve']})")

        # Final assessment
        result['macro_score'] = max(-1.0, min(1.0, macro_score))

        if macro_score > 0.3:
            result['net_liquidity_trend'] = 'EXPANDING'
            result['risk_bias'] = 'RISK_ON'
        elif macro_score < -0.3:
            result['net_liquidity_trend'] = 'CONTRACTING'
            result['risk_bias'] = 'RISK_OFF'
        else:
            result['net_liquidity_trend'] = 'STABLE'
            result['risk_bias'] = 'NEUTRAL'

        logger.info(f"Macro Assessment: {result['risk_bias']} (score={macro_score:+.2f}, "
                    f"Fed={result['fed_balance_trend']}, RRP={result['rrp_trend']}, "
                    f"Yield={result['yield_curve']})")

        return result

    def should_reduce_risk(self) -> Tuple[bool, str]:
        """
        Quick check: should we reduce position sizes due to macro conditions?
        
        Returns:
            Tuple of (reduce_risk: bool, reason: str)
        """
        assessment = self.get_liquidity_assessment()

        if assessment['yield_curve'] == 'INVERTED':
            return True, "Inverted yield curve - recession risk"

        if assessment['macro_score'] < -0.5:
            return True, f"Strong risk-off macro (score={assessment['macro_score']:.2f})"

        return False, "Macro conditions acceptable"

    def get_strength_modifier(self) -> float:
        """
        Get a signal strength modifier based on macro conditions.
        
        Returns:
            Float modifier: positive boosts signals, negative reduces them.
            Range: -0.15 to +0.15
        """
        assessment = self.get_liquidity_assessment()
        score = assessment['macro_score']

        # Map score (-1 to +1) to modifier (-0.15 to +0.15)
        return round(score * 0.15, 3)
