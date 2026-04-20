"""
Economic Calendar Filtering Module for Trading Bot

Fetches and filters economic events from ForexFactory public API to determine
if it's safe to trade specific currency pairs based on high-impact news events.

Handles timezone conversion (US Eastern to UTC) and tracks impact on specific
trading pairs (EUR_USD, XAU_USD, etc.).
"""

import requests
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsFilter:
    """
    Filters trading opportunities based on economic calendar events.

    Fetches events from ForexFactory public feed, caches results, and determines
    if high-impact events are scheduled within specified time windows for relevant
    currency pairs.
    """

    # Currency pair mappings to affected currencies
    PAIR_CURRENCY_MAP = {
        'EUR_USD': ['EUR', 'USD'],
        'XAU_USD': ['USD'],
        'GBP_USD': ['GBP', 'USD'],
        'USD_JPY': ['USD', 'JPY'],
        'USD_CHF': ['USD', 'CHF'],
        'AUD_USD': ['AUD', 'USD'],
        'NZD_USD': ['NZD', 'USD'],
        'CAD_USD': ['CAD', 'USD'],
    }

    # Impact level hierarchy
    IMPACT_LEVELS = {'High': 3, 'Medium': 2, 'Low': 1}

    # API endpoint - ForexFactory public calendar feed
    FF_API_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'

    # Cache duration in seconds (4 hours)
    CACHE_DURATION = 4 * 60 * 60

    # US Eastern timezone offset (used in the API data)
    US_EASTERN = timezone(timedelta(hours=-5))  # EST/EDT varies, but -5 is typical
    UTC = timezone.utc

    def __init__(
        self,
        pause_minutes_before: int = 30,
        pause_minutes_after: int = 15,
        min_impact: str = 'High'
    ):
        """
        Initialize NewsFilter.

        Args:
            pause_minutes_before: Minutes before event to consider unsafe (default 30)
            pause_minutes_after: Minutes after event to consider unsafe (default 15)
            min_impact: Minimum impact level to filter ('High', 'Medium', 'Low')

        Raises:
            ValueError: If min_impact is not a valid impact level
        """
        if min_impact not in self.IMPACT_LEVELS:
            raise ValueError(f"min_impact must be one of {list(self.IMPACT_LEVELS.keys())}")

        self.pause_minutes_before = pause_minutes_before
        self.pause_minutes_after = pause_minutes_after
        self.min_impact = min_impact
        self.min_impact_level = self.IMPACT_LEVELS[min_impact]

        # Cache storage
        self._calendar_cache: Optional[List[Dict]] = None
        self._cache_timestamp: Optional[datetime] = None

        logger.info(
            f"NewsFilter initialized: {pause_minutes_before}min before, "
            f"{pause_minutes_after}min after, min_impact={min_impact}"
        )

    def update_calendar(self) -> bool:
        """
        Fetch latest economic events from ForexFactory API.

        Uses cached data if available and less than 4 hours old.
        On API failure, preserves previous cache or defaults to empty list.

        Returns:
            bool: True if fresh data fetched, False if using cache or on error
        """
        # Check cache validity
        if self._calendar_cache is not None and self._cache_timestamp is not None:
            cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
            if cache_age < self.CACHE_DURATION:
                logger.debug(f"Using cached calendar (age: {cache_age:.0f}s)")
                return False

        try:
            logger.info(f"Fetching calendar from {self.FF_API_URL}")
            response = requests.get(self.FF_API_URL, timeout=10)
            response.raise_for_status()

            events = response.json()
            if not isinstance(events, list):
                logger.warning("API response is not a list, wrapping it")
                events = [events] if events else []

            self._calendar_cache = events
            self._cache_timestamp = datetime.now()
            logger.info(f"Successfully fetched {len(events)} events from calendar")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch calendar: {e}")
            # Preserve cache or default to empty
            if self._calendar_cache is None:
                self._calendar_cache = []
                logger.warning("No cache available, defaulting to empty calendar")
            else:
                logger.warning("Using stale cache due to API failure")
            return False
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse calendar JSON: {e}")
            if self._calendar_cache is None:
                self._calendar_cache = []
            return False

    def _parse_event_time(self, event: Dict) -> Optional[datetime]:
        """
        Parse event time from ForexFactory format to UTC datetime.

        ForexFactory format: date="04-11-2026", time="8:30am"
        Times are in US Eastern timezone.

        Args:
            event: Event dict from API

        Returns:
            datetime in UTC or None if parsing fails
        """
        try:
            date_str = event.get('date', '')
            time_str = event.get('time', '')

            if not date_str or not time_str:
                logger.debug(f"Missing date or time in event: {event.get('title', 'Unknown')}")
                return None

            # Parse date "04-11-2026" -> MM-DD-YYYY
            date_parts = date_str.split('-')
            if len(date_parts) != 3:
                logger.warning(f"Invalid date format: {date_str}")
                return None

            month, day, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])

            # Parse time "8:30am" or "8:30pm"
            time_str = time_str.strip().lower()
            is_pm = 'pm' in time_str
            time_str = time_str.replace('am', '').replace('pm', '').strip()

            time_parts = time_str.split(':')
            if len(time_parts) != 2:
                logger.warning(f"Invalid time format: {event.get('time')}")
                return None

            hour, minute = int(time_parts[0]), int(time_parts[1])

            # Convert to 24-hour format
            if is_pm and hour != 12:
                hour += 12
            elif not is_pm and hour == 12:
                hour = 0

            # Create datetime in US Eastern timezone
            event_time_eastern = datetime(
                year, month, day, hour, minute, 0,
                tzinfo=self.US_EASTERN
            )

            # Convert to UTC
            event_time_utc = event_time_eastern.astimezone(self.UTC)
            return event_time_utc

        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Failed to parse event time '{event.get('time')}': {e}")
            return None

    def _get_affected_currencies(self, event: Dict) -> List[str]:
        """
        Extract currencies affected by this event.

        Args:
            event: Event dict with 'country' field (3-letter code)

        Returns:
            List of currency codes affected (e.g., ['EUR', 'USD'])
        """
        country = event.get('country', '').upper().strip()

        # Map country codes to currency codes (usually same)
        # Handle special cases if needed
        if country in ['UK', 'GB']:
            return ['GBP']

        return [country] if country else []

    def _is_event_relevant(self, event: Dict, symbol: str) -> bool:
        """
        Check if event affects the given trading pair.

        Args:
            event: Event dict
            symbol: Trading pair (e.g., 'EUR_USD')

        Returns:
            True if event affects any currency in the pair
        """
        if symbol not in self.PAIR_CURRENCY_MAP:
            logger.warning(f"Unknown trading pair: {symbol}")
            return False

        pair_currencies = self.PAIR_CURRENCY_MAP[symbol]
        event_currencies = self._get_affected_currencies(event)

        # Check if any event currency matches pair currencies
        return any(ec in pair_currencies for ec in event_currencies)

    def _meets_impact_filter(self, event: Dict) -> bool:
        """
        Check if event meets minimum impact threshold.

        Args:
            event: Event dict with 'impact' field

        Returns:
            True if impact >= min_impact
        """
        impact = event.get('impact', 'Low').strip()
        event_level = self.IMPACT_LEVELS.get(impact, 0)
        return event_level >= self.min_impact_level

    def _get_minutes_until_event(self, event_time: datetime) -> float:
        """
        Calculate minutes from now until event.

        Args:
            event_time: Event time in UTC

        Returns:
            Minutes until event (negative if in the past)
        """
        now = datetime.now(self.UTC)
        delta = event_time - now
        return delta.total_seconds() / 60

    def is_safe_to_trade(self, symbol: str) -> Dict:
        """
        Determine if it's safe to trade a symbol based on upcoming news.

        Checks for high-impact events within the pause window.
        On API/parsing errors, defaults to safe=True (bot doesn't freeze).

        Args:
            symbol: Trading pair (e.g., 'EUR_USD', 'XAU_USD')

        Returns:
            Dict with keys:
                - 'safe': bool - True if no high-impact events nearby
                - 'reason': str - Explanation of safety status
                - 'next_event': str - Title of next relevant event or ''
                - 'minutes_until': float - Minutes until next event or None
        """
        # Fetch latest calendar
        self.update_calendar()

        if not self._calendar_cache:
            logger.warning("No calendar data available, defaulting to safe=True")
            return {
                'safe': True,
                'reason': 'No calendar data available',
                'next_event': '',
                'minutes_until': None
            }

        try:
            now = datetime.now(self.UTC)
            relevant_events = []

            # Parse and filter events
            for event in self._calendar_cache:
                # Skip if impact doesn't meet threshold
                if not self._meets_impact_filter(event):
                    continue

                # Skip if not relevant to this symbol
                if not self._is_event_relevant(event, symbol):
                    continue

                # Parse event time
                event_time = self._parse_event_time(event)
                if not event_time:
                    continue

                minutes_until = self._get_minutes_until_event(event_time)

                # Check if within pause window
                if -self.pause_minutes_after <= minutes_until <= self.pause_minutes_before:
                    relevant_events.append({
                        'title': event.get('title', 'Unknown'),
                        'time': event_time,
                        'minutes_until': minutes_until,
                        'impact': event.get('impact', 'Unknown'),
                        'country': event.get('country', 'Unknown')
                    })

            # Determine safety
            if relevant_events:
                # Sort by minutes_until to get nearest event
                relevant_events.sort(key=lambda x: abs(x['minutes_until']))
                nearest = relevant_events[0]

                title = nearest['title']
                minutes = nearest['minutes_until']

                if minutes >= 0:
                    reason = f"High-impact event in {minutes:.1f}min: {title}"
                else:
                    reason = f"High-impact event {abs(minutes):.1f}min ago: {title}"

                return {
                    'safe': False,
                    'reason': reason,
                    'next_event': title,
                    'minutes_until': minutes
                }
            else:
                # Find next relevant event for reporting
                next_event_title = ''
                next_minutes = None

                for event in self._calendar_cache:
                    if not self._is_event_relevant(event, symbol):
                        continue

                    event_time = self._parse_event_time(event)
                    if not event_time:
                        continue

                    minutes_until = self._get_minutes_until_event(event_time)

                    if minutes_until > self.pause_minutes_before:
                        if next_minutes is None or minutes_until < next_minutes:
                            next_event_title = event.get('title', 'Unknown')
                            next_minutes = minutes_until

                reason = "No high-impact events in pause window"
                return {
                    'safe': True,
                    'reason': reason,
                    'next_event': next_event_title,
                    'minutes_until': next_minutes
                }

        except Exception as e:
            logger.error(f"Error checking trade safety for {symbol}: {e}")
            # On any error, default to safe=True to prevent bot freezing
            return {
                'safe': True,
                'reason': 'Error checking calendar, defaulting to safe',
                'next_event': '',
                'minutes_until': None
            }

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """
        Get list of upcoming high-impact events for daily summary.

        Useful for Telegram alerts or dashboard displays.

        Args:
            hours_ahead: Look ahead this many hours (default 24)

        Returns:
            List of dicts with keys:
                - 'time': datetime in UTC
                - 'title': Event name
                - 'country': 3-letter country code
                - 'impact': 'High', 'Medium', 'Low'
                - 'forecast': Expected value
                - 'previous': Previous value
                - 'minutes_until': Minutes from now
        """
        self.update_calendar()

        if not self._calendar_cache:
            logger.warning("No calendar data for upcoming events")
            return []

        try:
            now = datetime.now(self.UTC)
            future_cutoff = now + timedelta(hours=hours_ahead)
            upcoming = []

            for event in self._calendar_cache:
                # Filter by impact
                if not self._meets_impact_filter(event):
                    continue

                # Parse time
                event_time = self._parse_event_time(event)
                if not event_time:
                    continue

                # Check if within time window
                if now <= event_time <= future_cutoff:
                    minutes_until = self._get_minutes_until_event(event_time)

                    upcoming.append({
                        'time': event_time,
                        'title': event.get('title', 'Unknown'),
                        'country': event.get('country', 'Unknown'),
                        'impact': event.get('impact', 'Unknown'),
                        'forecast': event.get('forecast', 'N/A'),
                        'previous': event.get('previous', 'N/A'),
                        'minutes_until': minutes_until
                    })

            # Sort by time
            upcoming.sort(key=lambda x: x['time'])
            logger.info(f"Found {len(upcoming)} upcoming events in next {hours_ahead}h")
            return upcoming

        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []


# Example usage and testing
if __name__ == '__main__':
    # Initialize filter
    nf = NewsFilter(pause_minutes_before=30, pause_minutes_after=15, min_impact='High')

    # Update calendar
    nf.update_calendar()

    # Check if safe to trade EUR_USD
    safety = nf.is_safe_to_trade('EUR_USD')
    print(f"\nEUR_USD Safety Status:")
    print(f"  Safe: {safety['safe']}")
    print(f"  Reason: {safety['reason']}")
    print(f"  Next Event: {safety['next_event']}")
    print(f"  Minutes Until: {safety['minutes_until']}")

    # Check XAU_USD
    safety = nf.is_safe_to_trade('XAU_USD')
    print(f"\nXAU_USD Safety Status:")
    print(f"  Safe: {safety['safe']}")
    print(f"  Reason: {safety['reason']}")
    print(f"  Next Event: {safety['next_event']}")
    print(f"  Minutes Until: {safety['minutes_until']}")

    # Get upcoming events
    events = nf.get_upcoming_events(hours_ahead=24)
    print(f"\nUpcoming High-Impact Events (next 24h): {len(events)}")
    for event in events[:5]:  # Show first 5
        print(f"  {event['time'].strftime('%H:%M UTC')} - {event['title']} ({event['country']})")
