from strategies.common import load_data
import click
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from strategies.common import load_data
import json
import os
from strategies.common import save_backtest_results
import time
import numpy as np


"""
two cases:
- bottom happens before cross.
- bottom happens after cross.

bottom detection:
- distance from peak.
- must be a triangle (in lows).
- must be smallest point you are looking at from current time.
- distance of candle leg (optional)
e.g. FLM 13nov 15:00
"""


class Chart(object):
    def __init__(self, data, from_period, until_period):
        self.data = data
        self.from_period = from_period
        self.until_period = until_period
        self.fig = plt.figure()
        _ = self.fig.add_subplot(1, 1, 1)

    def plot(self, save_path):
        plt.tight_layout()
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.savefig(save_path)
        plt.close()

    def closes(self):
        ax = self.fig.axes[0]
        ax.clear()
        times = []
        closes = []
        for i in range(self.from_period, self.until_period + 1):
            times.append(i)
            closes.append(self.data[i]['price_close'])
        ax.plot(times, closes, color='orange', markersize=5, marker='.', linewidth=0.5)
        ax.set_xticklabels([])

    def ohlc(self):
        ax = self.fig.axes[0]
        ax.clear()
        times = []
        highs = []
        lows = []
        closes = []
        for i in range(self.from_period, self.until_period + 1):
            times.append(i)
            highs.append(self.data[i]['price_high'])
            lows.append(self.data[i]['price_low'])
            closes.append(self.data[i]['price_close'])
        ax.plot(times, lows, color='blue', markersize=5, marker='.', linewidth=0.2)
        ax.plot(times, closes, color='orange', markersize=5, marker='.', linewidth=0.5)
        ax.plot(times, highs, color='blue', markersize=5, marker='.', linewidth=0.2)
        ax.set_xticklabels([])

    def kc(self):
        ax = self.fig.axes[0]
        times = []
        kcl = []
        kcb = []
        kcu = []
        for i in range(self.from_period, self.until_period + 1):
            times.append(i)
            kcl.append(self.data[i]['kcl'])
            kcb.append(self.data[i]['kcb'])
            kcu.append(self.data[i]['kcu'])
        ax.plot(times, kcl, color='red', linewidth=0.2)
        ax.plot(times, kcb, color='red', linewidth=0.5)
        ax.plot(times, kcu, color='red', linewidth=0.2)

    def sma(self, lag):
        ax = self.fig.axes[0]
        times = []
        ma = []
        for i in range(self.from_period, self.until_period + 1):
            times.append(i)
            ma.append(self.data[i]['sma_{}'.format(lag)])
        ax.plot(times, ma, color='blue', linewidth=1)

    def stoch(self):
        self.add_subplot()
        ax = self.fig.axes[-1]
        times = []
        fast = []
        slow = []
        for i in range(self.from_period, self.until_period + 1):
            times.append(i)
            fast.append(self.data[i]['stoch_fast'])
            slow.append(self.data[i]['stoch_slow'])
        ax.plot(times, fast, color='red', markersize=5, marker='.', linewidth=0.5)
        ax.plot(times, slow, color='blue', markersize=5, marker='.', linewidth=0.5)

    def enter(self, period_enter, price_enter, annotate=False, color=None, label=None):
        ax = self.fig.axes[0]
        ax.scatter([period_enter], [price_enter], marker='x', c=color, s=100, label=label)
        if annotate:
            ax.annotate('{}'.format(price_enter), color='magenta', xy=(period_enter, price_enter))

    def exit(self, period_exit, price_exit, annotate=False, color=None, label=None):
        ax = self.fig.axes[0]
        ax.scatter([period_exit], [price_exit], marker='x', c=color, s=100, label=label)
        if annotate:
            ax.annotate('{}'.format(price_exit), color='magenta', xy=(period_exit, price_exit))

    def custom(self, plot_data):
        for config in plot_data:
            if config['type'] == 'line_main':
                ax = self.fig.axes[0]
                ax.plot(config['xvals'], config['yvals'], color=config['color'], linewidth=0.2, marker='.', label=config['label'])
            if config['type'] == 'line_secondary':
                ax = self.fig.axes[1]
                ax.plot(config['xvals'], config['yvals'], color=config['color'], linewidth=0.2)
            if config['type'] == 'scatter':
                ax = self.fig.axes[0]
                xval = config['xval']
                yval = config['yval']
                label = config['label'] if 'label' in config else None
                color = config['color'] if 'color' in config else None
                ax.scatter([xval], [yval], marker='x', c=color, s=100, label=label)

    def add_subplot(self):
        n = len(self.fig.axes)
        for i in range(n):
            self.fig.axes[i].change_geometry(n + 1, 1, i + 1)
        _ = self.fig.add_subplot(n + 1, 1, n + 1)
        self.fig.axes[0].set_xticklabels([])


class IsTrendLargeDoubleSMA(object):
    def __init__(self, price_close, price_low, price_high, sma_short, sma_long, min_trend_width, min_trend_height,
                 offset):
        self.price_close = price_close
        self.price_low = price_low
        self.price_high = price_high
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.min_trend_width = min_trend_width
        self.min_trend_height = min_trend_height
        self.offset = offset
        self.is_uptrend = False
        self.curr_trend_width = 0
        self.last_trend_low = -1
        self._vals = np.zeros(len(price_close))
        self.init_vals()

    def __getitem__(self, t):
        return self._vals[t]

    def init_vals(self):
        t = self.get_first_upcross_period()
        t = t - 1
        for i in range(t):
            self._vals[t] = 0
        while True:
            if t == len(self.price_close):
                break
            while True:
                self.update_state(t)
                if not self.is_uptrend:
                    self._vals[t] = 0
                    break
                if not self.curr_trend_width >= self.min_trend_width:
                    self._vals[t] = 0
                    break
                # if uptrend and previous value 1 -> curr trend still wide & tall:
                if self._vals[t - 1] == 1:
                    self._vals[t] = 1
                    break
                # if trend both wide and tall:
                curr_trend_height = self.price_high[t] / self.last_trend_low
                if curr_trend_height >= self.min_trend_height:
                    self._vals[t] = 1
                    break
                self._vals[t] = 0
                break
            t += 1

    def get_first_upcross_period(self):
        # start is when first upcross happens.
        t0 = -1
        for i in range(len(self.price_close)):
            if isinstance(self.sma_long[i], float):
                t0 = i
                break
        for t in range(t0 + 1, len(self.price_close)):
            upcross = \
                self.sma_short[t] > self.sma_long[t] and \
                self.sma_short[t - 1] <= self.sma_long[t - 1]
            if upcross:
                return t
        return len(self.price_close)

    def update_state(self, t):
        upcross = \
                self.sma_short[t] > self.sma_long[t] and \
                self.sma_short[t - 1] <= self.sma_long[t - 1]
        if upcross:
            self.update_last_valley(until_period=t)
            self.is_uptrend = True
            self.curr_trend_width += 1
        elif self.sma_short[t] > self.sma_long[t]:
            self.is_uptrend = True
            self.curr_trend_width += 1
        else:
            self.is_uptrend = False
            self.curr_trend_width = 0

    def update_last_valley(self, until_period):
        from_period = max(until_period - self.offset, 0)
        min_val = self.price_low[from_period]
        for i in range(from_period + 1, until_period + 1):
            if self.price_low[i] < min_val:
                min_val = self.price_low[i]
        self.last_trend_low = min_val


class StratRetraceLong(object):
    # after SMA cross down of long up move you buy when it retraces back up.
    def __init__(self, data, sma_short_lag, sma_long_lag):
        self.data = data
        self.sma_short_lag = sma_short_lag
        self.sma_long_lag = sma_long_lag
        self.max_hold_time = 40
        self.sma_short = [self.data[t]['sma_{}'.format(self.sma_short_lag)] for t in range(len(self.data))]
        self.sma_long = [self.data[t]['sma_{}'.format(self.sma_long_lag)] for t in range(len(self.data))]
        price_close = [self.data[t]['price_close'] for t in range(len(self.data))]
        price_high = [self.data[t]['price_high'] for t in range(len(self.data))]
        price_low = [self.data[t]['price_high'] for t in range(len(self.data))]
        self.is_trend_large = IsTrendLargeDoubleSMA(
            price_close=price_close,
            price_low=price_low,
            price_high=price_high,
            sma_short=self.sma_short,
            sma_long=self.sma_long,
            min_trend_width=20,
            min_trend_height=1.01,
            offset=5
        )

    def get_exit(self, enter_period, exit_target, exit_stop):
        exit_period = None
        price_exit = None
        time_since_entry = 1
        for t in range(enter_period + 1, len(self.data)):
            if self.data[t]['price_low'] <= exit_stop:
                exit_period = t
                price_exit = exit_stop
                break
            if self.data[t]['price_high'] >= exit_target:
                exit_period = t
                price_exit = exit_target
                break
            if time_since_entry >= self.max_hold_time:
                exit_period = t
                price_exit = self.data[t]['price_close']
                break
            time_since_entry += 1
        if price_exit:
            output = {
                'exit_period': exit_period,
                'price_exit': price_exit
            }
            return output
        return None

    def get_max_between(self, from_period, until_period):
        best = self.data[from_period]['price_high']
        best_period = from_period
        for i in range(from_period, until_period + 1):
            if self.data[i]['price_high'] > best:
                best = self.data[i]['price_high']
                best_period = i
        return best, best_period

    def get_min_between(self, from_period, until_period):
        best = self.data[from_period]['price_low']
        best_period = from_period
        for i in range(from_period, until_period + 1):
            if self.data[i]['price_low'] < best:
                best = self.data[i]['price_low']
                best_period = i
        return best, best_period

    def detect_next_valley(self, from_period):
        for t in range(from_period, len(self.data)):
            curr_low_is_higher = self.data[t]['price_low'] > self.data[t - 1]['price_low']
            prev_low_is_lower = self.data[t - 1]['price_low']
            if curr_low_is_higher and prev_low_is_lower:
                valley = self.data[t - 1]['price_low']
                valley_period = t - 1
                output = {'valley': valley, 'valley_period': valley_period, 'valley_detect_period': t}
                return output
        return None

    def get_curr_uptrend_stats(self, from_period, sma_short, sma_long):
        # assumes we are currently in an uptrend.
        trend_start_period = None
        trend_end_period = None
        valley = None
        valley_period = None
        valley_detect_period = None
        for t in range(from_period, len(self.data)):
            if sma_short[t] <= sma_long[t]:
                trend_end_period = t
                break
        if not trend_end_period:
            return None
        for t in range(from_period, -1, -1):
            if sma_short[t -1] <= sma_long[t - 1] and sma_short[t] > sma_long[t]:
                trend_start_period = t
                break
        if not trend_start_period:
            return None
        peak, peak_period = self.get_max_between(trend_start_period, trend_end_period)
        minval, minval_period = self.get_min_between(peak_period, trend_end_period)
        if minval_period == trend_end_period:
            valley_stats = self.detect_next_valley(from_period=trend_end_period)
            if valley_stats:
                valley = valley_stats['valley']
                valley_period = valley_stats['valley_period']
                valley_detect_period = valley_stats['valley_detect_period']
        else:
            valley = minval
            valley_period = minval_period
            valley_detect_period = trend_end_period
        if not valley:
            return None
        output = {
            'trend_end_period': trend_end_period,
            'peak': peak,
            'peak_period': peak_period,
            'valley': valley,
            'valley_period': valley_period,
            'valley_detect_period': valley_detect_period
        }
        return output

    def get_entry(self, from_period):
        t = from_period - 1
        while True:
            if t > len(self.data) - 1:
                break
            while True:
                if not self.is_trend_large[t]:  # entry is after large up trend
                    break
                uptrend = self.get_curr_uptrend_stats(t, self.sma_short, self.sma_long)
                if not uptrend:
                    break
                # we are looking for when low happens after cross.
                if uptrend['valley_period'] < uptrend['trend_end_period']:
                    break

                vp = uptrend['valley_period']
                if self.data[vp]['price_close'] / self.data[vp]['price_low'] < 1.002:
                    break

                i = uptrend['valley_detect_period']
                if self.data[i]['price_low'] > self.data[i - 1]['price_low']:
                    entry_stats = {
                        'price_enter': self.data[i]['price_close'],
                        'enter_period': i,
                        'peak': uptrend['peak'],
                        'peak_period': uptrend['peak_period'],
                        'valley': uptrend['valley'],
                        'valley_period': uptrend['valley_period'],
                        'trend_end_period': uptrend['trend_end_period']
                    }
                    return entry_stats
            t += 1
        return None

    def collect(self):
        t = 1
        while True:
            if t > len(self.data) - 1:
                break

            entry_dict = self.get_entry(from_period=t)
            if not entry_dict:
                break

            price_enter = entry_dict['price_enter']
            enter_period = entry_dict['enter_period']
            valley = entry_dict['valley']
            exit_dict = self.get_exit(
                enter_period=enter_period,
                exit_target=price_enter * 1.02,
                exit_stop=valley * 1.0,
            )
            if not exit_dict:
                break
            exit_period = exit_dict['exit_period']
            price_exit = exit_dict['price_exit']
            gross_return = price_exit / price_enter

            plot_from_period = max(enter_period - 100, 0)
            plot_until_period = min(exit_period + 30, len(self.data) - 1)
            plot_data = [
                {
                    'type': 'scatter',
                    'label': 'valley',
                    'color': 'red',
                    'xval': entry_dict['valley_period'],
                    'yval': entry_dict['valley']
                },
                {
                    'type': 'scatter',
                    'label': 'peak',
                    'color': 'green',
                    'xval': entry_dict['peak_period'],
                    'yval': entry_dict['peak']
                }
            ]

            output = {
                'price_enter': price_enter,
                'enter_period': enter_period,
                'plot_from_period': plot_from_period,
                'plot_until_period': plot_until_period,
                'price_exit': price_exit,
                'exit_period': exit_period,
                'gross_return': gross_return,
                'plot_data': plot_data
            }

            t = exit_period + 1
            yield output


def run_backtest(sampler):
    cum_return = 1
    num_pos_trades = 0
    num_neg_trades = 0
    avg_return_per_trade = 0
    cum_returns = []
    backtest_stats = []
    trades = []
    k = 0
    for i, trade in enumerate(sampler.collect()):
        if not trade['gross_return']:
            continue
        trades.append(trade)
        backtest_stats.append({
            'trade': k,
            'price_enter': trade['price_enter'],
            'price_exit': trade['price_exit'],
            'gross_return': trade['gross_return'],
            'enter_period': trade['enter_period'],
            'exit_period': trade['exit_period']
        })
        cum_returns.append({'time': trade['exit_period'], 'return': cum_return})
        gross_return = trade['gross_return']
        avg_return_per_trade += gross_return
        hit_stop = False
        cum_return *= gross_return
        print('trade: {} \t return: {} \t cum_return: {} \t hit_stop: {}'.format(k, gross_return, cum_return, hit_stop))
        if gross_return > 1.002:
            num_pos_trades += 1
        else:
            num_neg_trades += 1
        k += 1
    num_trades = num_pos_trades + num_neg_trades
    avg_return_per_trade = avg_return_per_trade / num_trades
    summary_stats = [{
        'cum_return': cum_returns[-1]['return'],
        'avg_return_per_trade': avg_return_per_trade,
        'num_trades': num_trades,
        'num_pos_trades': num_pos_trades,
        'num_neg_trades': num_neg_trades,
    }]
    return trades, cum_returns, backtest_stats, summary_stats


def get_data(data_path, sma_short_lag, sma_long_lag):
    data_fast = load_data(data_path)
    df = pd.DataFrame.from_records(data_fast)
    df = df.rename(columns={
        'price_open': 'open',
        'price_high': 'high',
        'price_low': 'low',
        'price_close': 'close',
        'volume_traded': 'volume'
    })
    df.ta.sma(length=sma_short_lag, append=True)
    df.ta.sma(length=sma_long_lag, append=True)
    # df.fillna(value=-1, inplace=True)
    df = df.rename(columns={
        'open_x': 'price_open',
        'high_x': 'price_high',
        'low_x': 'price_low',
        'close_x': 'price_close',
        'volume_x': 'volume_traded',
        'SMA_{}'.format(sma_short_lag): 'sma_{}'.format(sma_short_lag),
        'SMA_{}'.format(sma_long_lag): 'sma_{}'.format(sma_long_lag)
    })
    df = df.rename(columns={
        'open': 'price_open',
        'high': 'price_high',
        'low': 'price_low',
        'close': 'price_close',
        'volume': 'volume_traded'
    })
    data = df.to_dict('records')
    return data


@click.command()
def main():
    sma_short_lag = 20
    sma_long_lag = 40

    run_dir = 'runs/retrace_long_{}'.format(time.time_ns() // 1000)
    os.makedirs(run_dir, exist_ok=True)
    data = get_data(
        data_path='data/binance_spot_eth_usdt_1min.json',
        sma_short_lag=sma_short_lag,
        sma_long_lag=sma_long_lag
    )
    sampler = StratRetraceLong(
        data=data,
        sma_short_lag=sma_short_lag,
        sma_long_lag=sma_long_lag
    )
    trades, cum_returns, backtest_stats, summary_stats = run_backtest(sampler)
    save_backtest_results(data, trades, cum_returns, backtest_stats, summary_stats, run_dir, save_trades=False)
    print(json.dumps(summary_stats, indent=4))

    os.makedirs('{}/plots/pos'.format(run_dir), exist_ok=True)
    os.makedirs('{}/plots/neg'.format(run_dir), exist_ok=True)

    for i, trade in enumerate(sampler.collect()):
        if i == 60:
            break
        chart = Chart(
            data=data,
            from_period=max(trade['enter_period'] - 60, 0),
            until_period=min(trade['exit_period'] + 60, len(data) - 1),
        )
        # chart.closes()
        chart.ohlc()
        chart.sma(sma_short_lag)
        chart.sma(sma_long_lag)
        chart.enter(trade['enter_period'], trade['price_enter'], annotate=True)
        chart.exit(trade['exit_period'], trade['price_exit'], annotate=True)
        chart.custom(trade['plot_data'])
        if trade['price_enter']:
            if trade['gross_return'] > 1.0:
                save_path = '{}/plots/pos/plot_{}.png'.format(run_dir, i)
            else:
                save_path = '{}/plots/neg/plot_{}.png'.format(run_dir, i)
            chart.plot(save_path)
            print('saved {}'.format(save_path), 'profit: ', trade['gross_return'],
                  'price_enter: ', trade['price_enter'], 'price_exit: ', trade['price_exit'])


if __name__ == '__main__':
    main()
