"""Diagnose why strategy generates so few signals."""
import sys
sys.path.insert(0, '.')

from config import OANDA_DEMO, ACTIVE_INSTRUMENTS, STRATEGY
from oanda_client import OandaClient
from indicators import compute_all_indicators

client = OandaClient(OANDA_DEMO)

for inst in ACTIVE_INSTRUMENTS:
    candles = client.get_candles(inst.symbol, inst.timeframe, count=5000,
                                 from_time='2025-10-01T00:00:00Z')
    if not candles:
        print(f'{inst.symbol}: No candles')
        continue

    print(f'\n=== {inst.symbol} ({len(candles)} candles) ===')
    ind = compute_all_indicators(candles, STRATEGY)
    if not ind:
        print('No indicators')
        continue

    st_dir = ind['supertrend_direction']
    ema_f = ind['ema_fast']
    ema_s = ind['ema_slow']
    adx_arr = ind['adx']
    macd_h = ind['macd_histogram']

    bull_flips = 0
    bear_flips = 0
    filt = {'session': 0, 'long_only': 0, 'ema': 0, 'adx': 0, 'macd': 0}
    passed = 0

    for i in range(1, len(st_dir)):
        prev_d = st_dir[i - 1]
        curr_d = st_dir[i]
        bull = prev_d <= 0 and curr_d > 0
        bear = prev_d >= 0 and curr_d < 0

        if not bull and not bear:
            continue

        if bull:
            bull_flips += 1
        if bear:
            bear_flips += 1

        blocked = False

        # Session filter
        t = candles[i].get('time', '')
        hour = int(t[11:13]) if len(t) >= 13 else 12
        if hour < 5 or hour > 18:
            filt['session'] += 1
            blocked = True

        # Long only
        if bear and inst.auto_long_only:
            filt['long_only'] += 1
            blocked = True

        # EMA
        if i < len(ema_f) and i < len(ema_s):
            if bull and ema_f[i] <= ema_s[i]:
                filt['ema'] += 1
                blocked = True
            if bear and ema_f[i] >= ema_s[i]:
                filt['ema'] += 1
                blocked = True

        # ADX
        if i < len(adx_arr):
            if adx_arr[i] < 18:
                filt['adx'] += 1
                blocked = True

        # MACD
        if i < len(macd_h):
            if bull and macd_h[i] <= 0:
                filt['macd'] += 1
                blocked = True
            if bear and macd_h[i] >= 0:
                filt['macd'] += 1
                blocked = True

        if not blocked:
            passed += 1

    total = bull_flips + bear_flips
    print(f'SuperTrend flips: {bull_flips} BULL + {bear_flips} BEAR = {total} total')
    print(f'Blocked by session (5-18 UTC):  {filt["session"]}')
    print(f'Blocked by long_only:           {filt["long_only"]}')
    print(f'Blocked by EMA alignment:       {filt["ema"]}')
    print(f'Blocked by ADX < 18:            {filt["adx"]}')
    print(f'Blocked by MACD histogram:      {filt["macd"]}')
    print(f'>>> PASSED ALL FILTERS:         {passed} out of {total}')
    print(f'(Note: one signal can be blocked by multiple filters)')
