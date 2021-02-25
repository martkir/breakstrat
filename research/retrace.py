import click
import matplotlib.pyplot as plt
from strategies.common import load_data
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
from strategies.common import save_backtest_results
import time
import json


class Chart(object):
    def __init__(self, data, plot_data):
        self.data = data
        self.plot_data = plot_data

    def plot(self, save_path):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for config in self.plot_data:
            if config['type'] == 'annotate':
                ax1.annotate(config['text'], color='magenta', xy=config['xy'])
            if config['type'] == 'scatter':
                if 'label' not in config:
                    config['label'] = None
                markersize = config['markersize'] if 'markersize' in config else 100
                ax1.scatter(config['xvals'], config['yvals'], marker='x', c=config['color'], s=markersize,
                            label=config['label'])
            if config['type'] == 'ohlc':
                from_period = config['from_period']
                until_period = config['until_period']
                times = []
                highs = []
                lows = []
                closes = []
                for i in range(from_period, until_period + 1):
                    times.append(i)
                    highs.append(self.data[i]['price_high'])
                    lows.append(self.data[i]['price_low'])
                    closes.append(self.data[i]['price_close'])
                ax1.plot(times, lows, color='blue', markersize=5, marker='.', linewidth=0.2)
                ax1.plot(times, closes, color='orange', markersize=5, marker='.', linewidth=0.5)
                ax1.plot(times, highs, color='blue', markersize=5, marker='.', linewidth=0.2)
            if config['type'] == 'line_main':
                ax1.plot(config['xvals'], config['yvals'], color=config['color'], linewidth=0.2, marker='.',
                         label=config['label'])
            if config['type'] == 'line_secondary':
                ax2.plot(config['xvals'], config['yvals'], color=config['color'], linewidth=0.2)

        ax1.legend()
        plt.savefig(save_path)
        plt.close()


def rename(df, original=False):
    if original:
        df = df.rename(columns={
            'open': 'price_open',
            'high': 'price_high',
            'low': 'price_low',
            'close': 'price_close',
            'volume': 'volume_traded'
        })
    else:
        df = df.rename(columns={
            'price_open': 'open',
            'price_high': 'high',
            'price_low': 'low',
            'price_close': 'close',
            'volume_traded': 'volume'
        })
    return df


class StratSMAShort(object):
    def __init__(self, data, sma_length, min_length=15, min_return=1.006, bound_coeff=0.35, exit_stop_coeff=1.001,
                 max_num_periods=60, exit_target_coeff=1.0):
        self.min_length = min_length
        self.min_return = min_return
        self.bound_coeff = bound_coeff
        self.exit_stop_coeff = exit_stop_coeff
        self.max_num_periods = max_num_periods
        self.exit_target_coeff = exit_target_coeff
        self.sma_length = sma_length
        df = pd.DataFrame.from_records(data)
        df = rename(df)
        df.ta.sma(length=sma_length, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df = rename(df, original=True)
        df = df.rename(columns={
            'SMA_{}'.format(sma_length): 'sma',
            'MACD_12_26_9': 'macd',
            'MACDh_12_26_9': 'macd_h',
            'MACDs_12_26_9': 'macd_s',
        })
        self.data = df.to_dict('records')

    def get_prev_up_cross_period(self, from_period):
        i = from_period
        while True:
            if i < 0:
                break
            is_up_cross = \
                self.data[i]['price_close'] > self.data[i]['sma'] and \
                self.data[i - 1]['price_close'] < self.data[i - 1]['sma']
            if is_up_cross:
                up_cross_period = i
                return up_cross_period
            else:
                i -= 1
        return 0

    def get_min_point(self, from_period, until_period):
        from_period = max(from_period, 0)
        min_val = self.data[from_period]['price_low']
        min_period = from_period
        for i in range(from_period + 1, until_period + 1):
            if self.data[i]['price_low'] < min_val:
                min_val = self.data[i]['price_low']
                min_period = i
        return min_val, min_period

    def get_max_point(self, from_period, until_period):
        from_period = max(from_period, 0)
        max_val = self.data[from_period]['price_high']
        max_period = from_period
        for i in range(from_period + 1, until_period + 1):
            if self.data[i]['price_high'] > max_val:
                max_val = self.data[i]['price_high']
                max_period = i
        return max_val, max_period

    def get_exit(self, enter_period, exit_target, exit_stop):
        exit_period = None
        price_exit = None

        time_since_entry = 1
        for t in range(enter_period + 1, len(self.data)):
            if self.data[t]['price_low'] <= exit_target:
                exit_period = t
                price_exit = exit_target
                break
            if self.data[t]['price_high'] >= exit_stop:
                exit_period = t
                price_exit = exit_stop
                break
            if time_since_entry >= self.max_num_periods:
                exit_period = t
                price_exit = self.data[t]['price_close']
                break
            time_since_entry += 1

        if price_exit:
            output = {
                'exit_period': exit_period,
                'price_exit': price_exit,
                'plot_data': [
                    {
                        'type': 'annotate',
                        'text': '{}'.format(price_exit),
                        'xy': (exit_period, price_exit)
                    },
                    {
                        'type': 'scatter',
                        'xvals': [exit_period],
                        'yvals': [price_exit],
                        'color': 'black',
                        'label': 'price_exit',
                    },
                    {
                        'type': 'line_main',
                        'xvals': [k for k in range(enter_period, exit_period + 1)],
                        'yvals': [exit_target] * (exit_period + 1 - enter_period),
                        'color': 'black',
                        'label': 'exit_target'
                    },
                    {
                        'type': 'line_main',
                        'xvals': [k for k in range(enter_period, exit_period + 1)],
                        'yvals': [exit_stop] * (exit_period + 1 - enter_period),
                        'color': 'magenta',
                        'label': 'exit_stop'
                    }
                ]
            }
            return output

        return None

    def get_entry(self, from_period):
        t = from_period
        while True:
            if t > len(self.data) - 1:
                break

            down_cross = \
                self.data[t]['price_close'] < self.data[t]['sma'] and \
                self.data[t - 1]['price_close'] > self.data[t - 1]['sma']

            prev_up_cross_period = self.get_prev_up_cross_period(from_period=t)
            num_periods_since_up_cross = t - prev_up_cross_period

            v0_value, v0_period = self.get_min_point(
                from_period=prev_up_cross_period - 10,
                until_period=prev_up_cross_period
            )
            p0_value, p0_period = self.get_max_point(
                from_period=prev_up_cross_period,
                until_period=t
            )

            up_move_is_long = num_periods_since_up_cross >= self.min_length
            up_move_is_large = p0_value / v0_value >= self.min_return

            entry_lower_bound = p0_value - self.bound_coeff * (p0_value - v0_value)
            price_is_close_to_peak = self.data[t]['price_close'] >= entry_lower_bound

            if down_cross and up_move_is_long and up_move_is_large and price_is_close_to_peak:
                enter_period = t
                price_enter = self.data[t]['price_close']
                output = {
                    'enter_period': enter_period,
                    'price_enter': price_enter,
                    'v0_value': v0_value,
                    'v0_period': v0_period,
                    'p0_value': p0_value,
                    'p0_period': p0_period,
                    'plot_data': [
                        {
                            'type': 'annotate',
                            'text': '{}'.format(price_enter),
                            'xy': (enter_period, price_enter)
                        },
                        {
                            'type': 'line_main',
                            'xvals': [k for k in range(p0_period, enter_period + 1)],
                            'yvals': [entry_lower_bound] * (enter_period + 1 - p0_period),
                            'color': 'red',
                            'label': 'entry_bound'
                        },
                        {
                            'type': 'scatter',
                            'xvals': [enter_period],
                            'yvals': [price_enter],
                            'label': 'price_enter',
                            'color': 'purple'
                        },
                        {
                            'type': 'scatter',
                            'xvals': [p0_period],
                            'yvals': [p0_value],
                            'label': 'peak',
                            'color': 'red'
                        },
                        {
                            'type': 'scatter',
                            'xvals': [v0_period],
                            'yvals': [v0_value],
                            'label': 'valley',
                            'color': 'red'
                        }
                    ]
                }
                return output

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

            v0_value = entry_dict['v0_value']
            v0_period = entry_dict['v0_period']
            p0_value = entry_dict['p0_value']
            enter_period = entry_dict['enter_period']

            exit_dict = self.get_exit(
                enter_period=enter_period,
                exit_target=p0_value - self.exit_target_coeff * (p0_value - v0_value),
                exit_stop=p0_value * self.exit_stop_coeff,
            )

            if not exit_dict:
                break

            exit_period = exit_dict['exit_period']
            price_enter = entry_dict['price_enter']
            price_exit = exit_dict['price_exit']

            gross_return = price_enter / price_exit

            plot_from_period = max(v0_period - 10, 0)
            plot_until_period = min(exit_period + 30, len(self.data) - 1)
            plot_times = [t for t in range(plot_from_period, plot_until_period + 1)]
            plot_data = [
                {
                    'type': 'ohlc',
                    'from_period': plot_from_period,
                    'until_period': plot_until_period
                },
                {
                    'type': 'line_secondary',
                    'xvals': plot_times,
                    'yvals': [self.data[i]['macd'] for i in range(plot_from_period, plot_until_period + 1)],
                    'color': 'magenta'
                },
                {
                    'type': 'line_secondary',
                    'xvals': plot_times,
                    'yvals': [self.data[i]['macd_s'] for i in range(plot_from_period, plot_until_period + 1)],
                    'color': 'red'
                },
            ]

            plot_data += entry_dict['plot_data']
            plot_data += exit_dict['plot_data']
            output = {
                'price_enter': price_enter,
                'signal_period': enter_period,
                'plot_until_period': plot_until_period,
                'price_exit': price_exit,
                'exit_period': exit_period,
                'gross_return': gross_return,
                'plot_data': plot_data
            }

            t = exit_period + 1
            yield output


def run_backtest(sampler, data_path):
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
            'enter_period': trade['signal_period'],
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
        # if with_plots:
        #     plot_trade(i, run_dir, trade)
    num_trades = num_pos_trades + num_neg_trades
    avg_return_per_trade = avg_return_per_trade / num_trades
    summary_stats = [{
        'data_path': data_path,
        'cum_return': cum_returns[-1]['return'],
        'avg_return_per_trade': avg_return_per_trade,
        'num_trades': num_trades,
        'num_pos_trades': num_pos_trades,
        'num_neg_trades': num_neg_trades,
    }]
    return trades, cum_returns, backtest_stats, summary_stats



@click.command()
@click.option('--data_path', type=str, default='data/binance_spot_eth_usdt_1min.json')
def main(data_path):
    run_dir = 'runs/retrace_{}'.format(time.time_ns() // 1000)
    os.makedirs(run_dir, exist_ok=True)
    data = load_data(data_path)
    os.makedirs('temp/pos', exist_ok=True)
    os.makedirs('temp/neg', exist_ok=True)

    # sampler = SamplerSMA(data, sma_length=20)
    sampler = StratSMAShort(
        data=data,
        sma_length=10,
        min_length=10,
        min_return=1.0075,
        bound_coeff=0.4,
        exit_stop_coeff=1.02,  # todo: if too low profit too high because exit price is wrong.
        max_num_periods=90,
        exit_target_coeff=0.75
    )

    trades, cum_returns, backtest_stats, summary_stats = run_backtest(sampler, data_path)
    save_backtest_results(data, trades, cum_returns, backtest_stats, summary_stats, run_dir, save_trades=False)
    print(json.dumps(summary_stats, indent=4))

    for i, trade in enumerate(sampler.collect()):
        if i == 30:
            break
        chart = Chart(data, plot_data=trade['plot_data'])
        if trade['price_enter']:
            if trade['gross_return'] > 1.0:
                save_path = 'temp/pos/plot_{}.png'.format(i)
            else:
                save_path = 'temp/neg/plot_{}.png'.format(i)
            chart.plot(save_path)
            print('saved {}'.format(save_path), 'profit: ', trade['gross_return'],
                  'price_enter: ', trade['price_enter'], 'price_exit: ', trade['price_exit'])


if __name__ == '__main__':
    main()

