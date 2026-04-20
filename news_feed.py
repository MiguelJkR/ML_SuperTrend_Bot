"""
ML SuperTrend v51 - Feed de Noticias Financieras en Vivo (Gratuito)
====================================================================
Obtiene noticias financieras en tiempo real de fuentes 100% gratuitas
y las procesa con el SentimentEngine para generar señales.

Fuentes Gratuitas:
  1. RSS Feeds — Reuters, CNBC, MarketWatch, Investing.com (ilimitado)
  2. NewsAPI.org — Free tier: 100 requests/día, últimas 24h
  3. Finnhub — Free tier: 60 calls/min, noticias de mercado

Flujo:
  News Feed → Filtro por relevancia → SentimentEngine → score por instrumento
  → Feature para LSTM (sentiment_score como input adicional)

Uso:
    feed = NewsFeed()
    news = feed.get_forex_news()
    sentiment = feed.get_instrument_sentiment("EUR_USD")
    # → {"score": 0.25, "headlines": 5, "trend": "positive"}
"""

import logging
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # Gratis: https://newsapi.org/register
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")  # Gratis: https://finnhub.io/register
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_cache")
NEWS_TTL_MINUTES = 30  # Refrescar noticias cada 30 min

# RSS Feeds gratuitos (no requieren API key)
RSS_FEEDS = {
    "reuters_markets": "https://news.google.com/rss/search?q=forex+market&hl=en-US&gl=US&ceid=US:en",
    "forex_factory": "https://news.google.com/rss/search?q=forex+trading+currency&hl=en-US&gl=US&ceid=US:en",
    "cnbc_finance": "https://news.google.com/rss/search?q=financial+markets+stocks&hl=en-US&gl=US&ceid=US:en",
    "crypto_news": "https://news.google.com/rss/search?q=bitcoin+cryptocurrency&hl=en-US&gl=US&ceid=US:en",
    "gold_oil": "https://news.google.com/rss/search?q=gold+oil+commodities+price&hl=en-US&gl=US&ceid=US:en",
}

# Palabras clave por instrumento para filtrar relevancia
INSTRUMENT_KEYWORDS = {
    "EUR_USD": ["euro", "eur", "ecb", "european central bank", "eurozone", "eu economy",
                "dollar", "usd", "fed", "federal reserve", "eurusd"],
    "GBP_USD": ["pound", "gbp", "sterling", "bank of england", "boe", "uk economy",
                "brexit", "gbpusd"],
    "USD_JPY": ["yen", "jpy", "bank of japan", "boj", "japan economy", "usdjpy",
                "nikkei"],
    "USD_CHF": ["franc", "chf", "swiss", "snb", "switzerland", "usdchf"],
    "AUD_USD": ["australian", "aud", "rba", "reserve bank australia", "audusd"],
    "NZD_USD": ["zealand", "nzd", "rbnz", "nzdusd", "kiwi"],
    "USD_CAD": ["canadian", "cad", "boc", "bank of canada", "usdcad", "loonie"],
    "XAU_USD": ["gold", "xau", "precious metal", "safe haven", "bullion"],
    "BTC_USDT": ["bitcoin", "btc", "crypto", "cryptocurrency", "blockchain"],
    "ETH_USDT": ["ethereum", "eth", "crypto", "defi", "smart contract"],
}

# Keywords globales que afectan a todos los pares
GLOBAL_KEYWORDS = [
    "federal reserve", "fed", "interest rate", "inflation", "cpi",
    "gdp", "unemployment", "jobs report", "nonfarm", "trade war",
    "recession", "stimulus", "quantitative", "tapering", "dovish",
    "hawkish", "risk", "volatility", "geopolitical",
]


class NewsFeed:
    """
    Feed de noticias financieras en tiempo real.
    Combina múltiples fuentes gratuitas.
    """

    def __init__(self, sentiment_engine=None):
        self._sentiment = sentiment_engine
        self._cache: Dict[str, Dict] = {}
        self._last_fetch: Dict[str, float] = {}
        self._all_headlines: List[Dict] = []

        os.makedirs(CACHE_DIR, exist_ok=True)

        # Intentar importar SentimentEngine si no se proporcionó
        if self._sentiment is None:
            try:
                from sentiment_engine import SentimentEngine
                self._sentiment = SentimentEngine()
                logger.info("SentimentEngine cargado internamente")
            except Exception as e:
                logger.warning(f"SentimentEngine no disponible: {e}")

    # ─────────────────────────────────────────────────────────
    # RSS FEEDS (100% gratis, sin límite)
    # ─────────────────────────────────────────────────────────
    def fetch_rss_news(self) -> List[Dict]:
        """Obtener noticias de RSS feeds (Google News RSS)."""
        cache_key = "rss_news"
        cached = self._get_timed_cache(cache_key, NEWS_TTL_MINUTES)
        if cached:
            return cached

        all_news = []

        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                req = urllib.request.Request(feed_url, headers={
                    'User-Agent': 'Mozilla/5.0 (ML SuperTrend Bot)'
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    xml_data = resp.read().decode('utf-8', errors='ignore')

                root = ET.fromstring(xml_data)
                channel = root.find('channel')
                if channel is None:
                    continue

                items = channel.findall('item')
                for item in items[:10]:  # Top 10 per feed
                    title = item.findtext('title', '').strip()
                    link = item.findtext('link', '')
                    pub_date = item.findtext('pubDate', '')
                    description = item.findtext('description', '').strip()

                    if title:
                        all_news.append({
                            "title": title,
                            "source": feed_name,
                            "link": link,
                            "published": pub_date,
                            "description": description[:200],
                        })

            except Exception as e:
                logger.debug(f"RSS {feed_name} error: {e}")

        self._set_timed_cache(cache_key, all_news)
        self._all_headlines = all_news
        return all_news

    # ─────────────────────────────────────────────────────────
    # NEWSAPI (gratis: 100 req/día)
    # ─────────────────────────────────────────────────────────
    def fetch_newsapi(self, query: str = "forex OR currency OR federal reserve") -> List[Dict]:
        """Obtener noticias de NewsAPI.org (free tier)."""
        if not NEWSAPI_KEY:
            return []

        cache_key = f"newsapi_{query[:20]}"
        cached = self._get_timed_cache(cache_key, 60)  # Cache 1 hora
        if cached:
            return cached

        try:
            params = urllib.parse.urlencode({
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": NEWSAPI_KEY,
            })
            url = f"https://newsapi.org/v2/everything?{params}"
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            articles = data.get("articles", [])
            news = []
            for a in articles:
                title = a.get("title", "").strip()
                if title and title != "[Removed]":
                    news.append({
                        "title": title,
                        "source": a.get("source", {}).get("name", "NewsAPI"),
                        "link": a.get("url", ""),
                        "published": a.get("publishedAt", ""),
                        "description": (a.get("description") or "")[:200],
                    })

            self._set_timed_cache(cache_key, news)
            return news

        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # FINNHUB (gratis: 60 calls/min)
    # ─────────────────────────────────────────────────────────
    def fetch_finnhub_news(self, category: str = "forex") -> List[Dict]:
        """Obtener noticias de Finnhub (free tier)."""
        if not FINNHUB_KEY:
            return []

        cache_key = f"finnhub_{category}"
        cached = self._get_timed_cache(cache_key, NEWS_TTL_MINUTES)
        if cached:
            return cached

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            url = (f"https://finnhub.io/api/v1/news?"
                   f"category={category}&token={FINNHUB_KEY}")
            req = urllib.request.Request(url)

            with urllib.request.urlopen(req, timeout=10) as resp:
                articles = json.loads(resp.read().decode('utf-8'))

            news = []
            for a in articles[:20]:
                title = a.get("headline", "").strip()
                if title:
                    news.append({
                        "title": title,
                        "source": a.get("source", "Finnhub"),
                        "link": a.get("url", ""),
                        "published": datetime.fromtimestamp(
                            a.get("datetime", 0), tz=timezone.utc
                        ).isoformat() if a.get("datetime") else "",
                        "description": (a.get("summary") or "")[:200],
                    })

            self._set_timed_cache(cache_key, news)
            return news

        except Exception as e:
            logger.warning(f"Finnhub error: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # ANÁLISIS POR INSTRUMENTO
    # ─────────────────────────────────────────────────────────
    def get_all_news(self) -> List[Dict]:
        """Obtener todas las noticias de todas las fuentes."""
        all_news = []
        all_news.extend(self.fetch_rss_news())
        all_news.extend(self.fetch_newsapi())
        all_news.extend(self.fetch_finnhub_news())

        # Deduplicar por título similar
        seen = set()
        unique = []
        for n in all_news:
            key = n["title"][:50].lower()
            if key not in seen:
                seen.add(key)
                unique.append(n)

        self._all_headlines = unique
        return unique

    def filter_by_instrument(self, instrument: str, news: List[Dict] = None) -> List[Dict]:
        """Filtrar noticias relevantes para un instrumento específico."""
        if news is None:
            news = self._all_headlines or self.get_all_news()

        keywords = INSTRUMENT_KEYWORDS.get(instrument, []) + GLOBAL_KEYWORDS
        relevant = []

        for n in news:
            text = (n["title"] + " " + n.get("description", "")).lower()
            for kw in keywords:
                if kw.lower() in text:
                    n["relevance_keyword"] = kw
                    relevant.append(n)
                    break

        return relevant

    def get_instrument_sentiment(self, instrument: str) -> Dict:
        """
        Obtener sentimiento actual para un instrumento específico.
        Combina noticias filtradas + análisis de sentimiento.

        Returns: {
            "score": float (-1 a +1),
            "label": str,
            "headline_count": int,
            "positive_pct": float,
            "negative_pct": float,
            "trend": str ("improving"/"declining"/"stable"),
            "top_headlines": [{"title", "score", "source"}],
        }
        """
        cache_key = f"sentiment_{instrument}"
        cached = self._get_timed_cache(cache_key, NEWS_TTL_MINUTES)
        if cached:
            return cached

        relevant = self.filter_by_instrument(instrument)

        if not relevant or not self._sentiment:
            return {
                "score": 0.0, "label": "neutral", "headline_count": 0,
                "positive_pct": 0, "negative_pct": 0, "trend": "stable",
                "top_headlines": [],
            }

        # Analizar sentimiento de cada titular
        headlines = [n["title"] for n in relevant[:20]]
        results = self._sentiment.analyze_batch(headlines)

        scored_headlines = []
        for news_item, result in zip(relevant[:20], results):
            scored_headlines.append({
                "title": news_item["title"][:100],
                "score": result["score"],
                "label": result["label"],
                "source": news_item.get("source", "?"),
            })

        # Calcular score compuesto
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        labels = [r["label"] for r in results]
        n = len(results)
        pos_pct = sum(1 for l in labels if l == "positive") / n * 100 if n else 0
        neg_pct = sum(1 for l in labels if l == "negative") / n * 100 if n else 0

        # Trend: comparar primera mitad con segunda mitad
        if len(scores) >= 4:
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            if second_half > first_half + 0.05:
                trend = "improving"
            elif second_half < first_half - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        result = {
            "score": round(avg_score, 3),
            "label": "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral",
            "headline_count": len(relevant),
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "trend": trend,
            "top_headlines": sorted(scored_headlines, key=lambda x: abs(x["score"]), reverse=True)[:5],
        }

        self._set_timed_cache(cache_key, result)
        return result

    def get_market_overview(self) -> Dict:
        """
        Resumen general del sentimiento del mercado.
        """
        all_news = self.get_all_news()

        if not all_news or not self._sentiment:
            return {"available": False, "news_count": 0}

        headlines = [n["title"] for n in all_news[:30]]
        market_sentiment = self._sentiment.get_market_sentiment(headlines)

        return {
            "available": True,
            "news_count": len(all_news),
            **market_sentiment,
            "sources": list(set(n.get("source", "?") for n in all_news)),
        }

    # ─────────────────────────────────────────────────────────
    # CACHE
    # ─────────────────────────────────────────────────────────
    def _get_timed_cache(self, key: str, ttl_minutes: float) -> Optional:
        """Cache con TTL en minutos."""
        if key in self._cache:
            age_min = (time.time() - self._last_fetch.get(key, 0)) / 60
            if age_min < ttl_minutes:
                return self._cache[key]
        return None

    def _set_timed_cache(self, key: str, data):
        """Guardar en cache."""
        self._cache[key] = data
        self._last_fetch[key] = time.time()

    def get_status(self) -> Dict:
        """Estado del feed para dashboard."""
        return {
            "rss_available": True,
            "newsapi_available": bool(NEWSAPI_KEY),
            "finnhub_available": bool(FINNHUB_KEY),
            "sentiment_engine": self._sentiment.get_status() if self._sentiment else {"backend": "none"},
            "cached_headlines": len(self._all_headlines),
            "cache_keys": len(self._cache),
            "sources": list(RSS_FEEDS.keys()),
        }


if __name__ == "__main__":
    print("=== News Feed Test ===\n")
    feed = NewsFeed()
    print("Status:", json.dumps(feed.get_status(), indent=2))

    print("\nObteniendo noticias RSS...")
    news = feed.fetch_rss_news()
    print(f"  {len(news)} titulares obtenidos")
    for n in news[:5]:
        print(f"  • [{n['source']}] {n['title'][:80]}")

    print("\nSentimiento EUR_USD:")
    sent = feed.get_instrument_sentiment("EUR_USD")
    print(f"  Score: {sent['score']:+.3f} ({sent['label']})")
    print(f"  Headlines relevantes: {sent['headline_count']}")
    for h in sent.get("top_headlines", [])[:3]:
        emoji = "🟢" if h["score"] > 0 else "🔴" if h["score"] < 0 else "⚪"
        print(f"  {emoji} [{h['score']:+.3f}] {h['title'][:70]}")
