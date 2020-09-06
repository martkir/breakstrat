import click
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import time
import pandas as pd


plt.rcParams["figure.figsize"] = (16, 8)


def is_breakout(hits):
    prices = [hit['price_close'] for hit in hits]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    max_abs_past_return = np.max(np.abs(returns[:len(returns) - 1] - 1))
    last_return = returns[-1] - 1
    if last_return / max_abs_past_return > 1.005:
        return True
    else:
        return False


class Chart(object):
    def __init__(self, candles, stop_levels, price_enter, price_exit, start_period, enter_period, exit_period, dirname):
        self.dirname = dirname
        self.candles = candles
        self.price_enter = price_enter
        self.price_exit = price_exit
        self.start_period = start_period
        self.enter_period = enter_period
        self.exit_period = exit_period
        self.stop_levels = stop_levels
        os.makedirs(self.dirname, exist_ok=True)

    def save(self):
        times = []
        highs = []
        lows = []
        closes = []
        for i in range(len(self.candles)):
            times.append(i + self.start_period)
            highs.append(self.candles[i]['price_high'])
            lows.append(self.candles[i]['price_low'])
            closes.append(self.candles[i]['price_close'])
        plt.figure()
        plt.plot(times, lows, color='blue', markersize=5, marker='.', linewidth=0.2)
        plt.plot(times, closes, color='orange', markersize=5, marker='.', linewidth=0.5)
        plt.plot(times, highs, color='blue', markersize=5, marker='.', linewidth=0.2)
        plt.plot(times, self.stop_levels, color='red', markersize=5, marker='.', linewidth=0.2)
        plt.plot([self.enter_period], [self.price_enter], 'x', markersize=5, color='magenta')
        plt.annotate('{}'.format(self.price_enter), color='magenta', xy=(self.enter_period, self.price_enter * 0.998))
        plt.savefig('{}/plot.png'.format(self.dirname))
        plt.close()


def plot_trade(trade_cnt, run_dir, trade_stats):
    chart_params = {
        'candles': trade_stats['candles'],
        'stop_levels': trade_stats['stop_levels'],
        'price_enter': trade_stats['price_enter'],
        'price_exit': trade_stats['price_exit'],
        'start_period': trade_stats['start_period'],
        'enter_period': trade_stats['signal_period'],
        'exit_period': trade_stats['exit_period'],
    }
    chart_pos_dir = os.path.join(run_dir, 'plots/pos')
    chart_neg_dir = os.path.join(run_dir, 'plots/neg')
    os.makedirs(chart_pos_dir, exist_ok=True)
    os.makedirs(chart_neg_dir, exist_ok=True)
    if trade_stats['gross_return'] > 1.0:
        chart_pos = Chart(**chart_params, dirname=os.path.join(chart_pos_dir, '{}'.format(trade_cnt)))
        chart_pos.save()
    else:
        chart_neg = Chart(**chart_params, dirname=os.path.join(chart_neg_dir, '{}'.format(trade_cnt)))
        chart_neg.save()


def plot_equity_curve(cum_returns, run_dir):
    times = [i for i in range(len(cum_returns))]
    plt.plot(times, cum_returns)
    plt.savefig(os.path.join(run_dir, 'equity.png'))


def load_data(data_path):
    if not os.path.exists(data_path):
        ValueError('Path {} does not exist.'.format(data_path))
    data = json.load(open(data_path, 'r'))
    print('--- Finished loading data at {} into memory ---'.format(data_path))
    return data


def trade_sampler(data, lookback):
    stop_coeff_initial = 0.985
    stop_coeff = 0.99
    target_coeff = 1.05 * 4
    terminal_num_periods = 20
    i = lookback

    while True:
        if i > len(data) - 1:
            break
        segment = data[i - lookback: i + 1]

        if is_breakout(segment):
            stop_levels = [None] * len(segment)
            candles = segment

            price_enter = data[i]['price_close']
            current_high = data[i]['price_high']
            stop_level = stop_coeff_initial * current_high

            target_level = target_coeff * price_enter
            hit_time_limit = False
            hit_stop = False
            time_since_entry = 1
            signal_period = i
            start_period = i - lookback
            i += 1

            while True:
                candles.append(data[i])
                stop_levels.append(stop_level)

                if data[i]['price_low'] <= stop_level:
                    price_exit = stop_level
                    exit_period = i
                    gross_return = price_exit / price_enter
                    hit_stop = True
                    break

                if data[i]['price_high'] >= target_level:
                    price_exit = target_level
                    exit_period = i
                    gross_return = price_exit / price_enter
                    break

                if time_since_entry == terminal_num_periods:
                    price_exit = data[i]['price_close']
                    exit_period = i
                    gross_return = price_exit / price_enter
                    hit_time_limit = True
                    break

                if current_high < data[i]['price_high']:
                    current_high = data[i]['price_high']

                stop_level = current_high * stop_coeff
                i += 1
                time_since_entry += 1

            output = {
                'candles': candles,
                'gross_return': gross_return,
                'hit_time_limit': hit_time_limit,
                'hit_stop': hit_stop,
                'price_enter': price_enter,
                'price_exit': price_exit,
                'start_period': start_period,
                'exit_period': exit_period,
                'signal_period': signal_period,
                'stop_levels': stop_levels
            }

            yield output
        else:
            i += 1


@click.command()
@click.option('--with_plots', is_flag=True)
@click.option('--data_path', type=str, default='data/binance_spot_eth_usdt_5min.json')
@click.option('--stop_coeff_initial', type=str, default=0.985)
@click.option('--stop_coeff', type=str, default=0.99)
@click.option('--target_coeff', type=str, default=1.15)
@click.option('--terminal_num_periods', type=str, default=20)
@click.option('--lookback', type=int, default=60)
def main(with_plots, data_path, stop_coeff_initial, stop_coeff, target_coeff, terminal_num_periods, lookback):
    run_dir = 'runs/trailstop_{}'.format(time.time_ns() // 1000)
    strategy_params = {
        'data_path': data_path,
        'stop_coeff_initial': stop_coeff_initial,
        'stop_coeff': stop_coeff,
        'target_coeff': target_coeff,
        'terminal_num_periods': terminal_num_periods
    }
    os.makedirs(run_dir, exist_ok=True)
    json.dump(strategy_params, open(os.path.join(run_dir, 'params.json'), 'w+'), indent=4)

    backtest_stats = []
    data = load_data(data_path)
    cum_returns = []
    cum_return = 1
    num_pos_trades = 0
    num_neg_trades = 0

    for i, trade_stats in enumerate(trade_sampler(data, lookback=60)):
        backtest_stats.append({
            'trade': i,
            'price_enter': trade_stats['price_enter'],
            'price_exit': trade_stats['price_exit'],
            'gross_return': trade_stats['gross_return'],
            'enter_period': trade_stats['signal_period'],
            'exit_period': trade_stats['exit_period']
        })

        cum_returns.append(cum_return)
        gross_return = trade_stats['gross_return']
        hit_stop = trade_stats['hit_stop']
        cum_return *= gross_return

        print('trade: {} \t return: {} \t cum_return: {} \t hit_stop: {}'.format(i, gross_return, cum_return, hit_stop))

        if gross_return > 1.0:
            num_pos_trades += 1
        else:
            num_neg_trades += 1

        if with_plots:
            plot_trade(i, run_dir, trade_stats)

    plot_equity_curve(cum_returns, run_dir)

    num_trades = num_pos_trades + num_neg_trades
    summary_stats = [{
        'num_trades': num_trades,
        'num_pos_trades': num_pos_trades,
        'num_neg_trades': num_neg_trades,
    }]
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(os.path.join(run_dir, 'summary.csv'), index=False, mode='w+')
    df_backtest_stats = pd.DataFrame(backtest_stats)
    df_backtest_stats.to_csv(os.path.join(run_dir, 'trades.csv'), index=False, mode='w+')

    print('--- Finished backtest. Results saved in {} ---'.format(run_dir))


if __name__ == '__main__':
    main()