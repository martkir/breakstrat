import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["figure.figsize"] = (16, 8)


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
        plt.xticks(list(plt.xticks()[0]) + [self.enter_period])
        plt.annotate('{}'.format(self.price_enter), color='magenta', xy=(self.enter_period, self.price_enter * 0.998))
        plt.savefig('{}/plot.png'.format(self.dirname))
        plt.close()


def get_cum_returns_buyhold(data):
    cum_returns = [{'time': 0, 'return': 1.0}]
    for i in range(1, len(data)):
        r = data[i]['price_close'] / data[i - 1]['price_close']
        cum_returns.append({'time': i, 'return': r * cum_returns[-1]['return']})
    return cum_returns


def plot_equity_curve(cum_returns, cum_returns_buyhold, run_dir):
    n = len(cum_returns)
    m = len(cum_returns_buyhold)
    times_strat = [cum_returns[i]['time'] for i in range(n)]
    times_buyhold = [cum_returns_buyhold[i]['time'] for i in range(m)]
    returns_strat = [cum_returns[i]['return'] for i in range(n)]
    returns_buyhold = [cum_returns_buyhold[i]['return'] for i in range(m)]
    plt.plot(times_strat, returns_strat, linewidth=0.5, color="blue", label="baseline")
    plt.plot(times_buyhold, returns_buyhold, linewidth=0.5, color="magenta", label="buy & hold")
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'equity.png'))


def save_backtest_results(data, trades, cum_returns, backtest_stats, summary_stats, run_dir, save_trades):
    if save_trades:
        json.dump(trades, open(os.path.join(run_dir, 'trades.json'), 'w+'), indent=4)
    cum_returns_buyhold = get_cum_returns_buyhold(data)
    plot_equity_curve(cum_returns, cum_returns_buyhold, run_dir)
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(os.path.join(run_dir, 'summary.csv'), index=False, mode='w+')
    df_summary.to_markdown(open(os.path.join(run_dir, 'summary.md'), 'w+'), index=False)
    df_backtest_stats = pd.DataFrame(backtest_stats)
    df_backtest_stats.to_csv(os.path.join(run_dir, 'trades.csv'), index=False, mode='w+')
    print('--- Finished backtest. Results saved in {} ---'.format(run_dir))


def run_backtest(trade_sampler, data_path, trade_params, with_plots, run_dir):
    data = load_data(data_path)
    cum_return = 1
    num_pos_trades = 0
    num_neg_trades = 0
    avg_return_per_trade = 0
    cum_returns = []
    backtest_stats = []
    trades = []
    for i, trade in enumerate(trade_sampler(data, trade_params)):
        trades.append(trade)
        backtest_stats.append({
            'trade': i,
            'price_enter': trade['price_enter'],
            'price_exit': trade['price_exit'],
            'gross_return': trade['gross_return'],
            'enter_period': trade['signal_period'],
            'exit_period': trade['exit_period']
        })
        cum_returns.append({'time': trade['exit_period'], 'return': cum_return})
        gross_return = trade['gross_return']
        avg_return_per_trade += gross_return
        hit_stop = trade['hit_stop']
        cum_return *= gross_return
        print('trade: {} \t return: {} \t cum_return: {} \t hit_stop: {}'.format(i, gross_return, cum_return, hit_stop))
        if gross_return > 1.002:
            num_pos_trades += 1
        else:
            num_neg_trades += 1
        if with_plots:
            plot_trade(i, run_dir, trade)
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
    return data, trades, cum_returns, backtest_stats, summary_stats


def is_breakout(candles, params, method):
    if method == 'abs':
        return is_breakout_abs(candles, params['lb_breakout'])
    elif method == 'sdev':
        return is_breakout_sdev(candles, params['alpha'], params['beta'])
    else:
        raise ValueError('Method {} is not valid.'.format(method))


def is_breakout_sdev(candles, alpha=0.5, beta=2.0):
    prices = [candle['price_close'] for candle in candles]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    r_log = np.log(returns)
    sigma = alpha * np.max(np.abs(r_log[:len(returns) - 1]))
    if np.abs(r_log[-1]) > beta * sigma and returns[-1] > 1.0:
        return True
    else:
        return False


def is_breakout_abs(candles, lb_breakout):
    prices = [candle['price_close'] for candle in candles]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    max_abs_past_return = np.max(np.abs(returns[:len(returns) - 1] - 1))
    last_return = returns[-1] - 1
    if last_return / max_abs_past_return > lb_breakout:
        return True
    else:
        return False


def calc_price_enter(candles, params, method):
    if method == 'abs':
        return calc_price_enter_abs(candles, params['lb_breakout'])
    elif method == 'sdev':
        return calc_price_enter_sdev(candles, params['alpha'], params['beta'])
    else:
        raise ValueError('Method {} is not valid.'.format(method))


def calc_price_enter_sdev(candles, alpha=0.5, beta=2.0):
    prices = [candle['price_close'] for candle in candles]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    r_log = np.log(returns)
    sigma = alpha * np.max(np.abs(r_log[:len(returns) - 1]))
    last_return = np.exp(beta * sigma)
    price_enter = prices[len(prices) - 2] * last_return
    return price_enter


def calc_price_enter_abs(candles, lb_breakout):
    prices = [candle['price_close'] for candle in candles]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    max_abs_past_return = np.max(np.abs(returns[:len(returns) - 1] - 1))  # -1 s.t. negative moves counted.
    last_return = lb_breakout * max_abs_past_return + 1  # min return needed.
    price_enter = prices[len(prices) - 2] * last_return
    return price_enter


def get_log_returns(candles):
    r_log = np.array([None] * len(candles))
    for i in range(1, len(candles)):
        r_log[i] = np.log(candles[i]['price_close'] / candles[i - 1]['price_close'])
    return r_log


def get_price_diffs(candles):
    p_diff = np.array([None] * len(candles))
    for i in range(1, len(candles)):
        p_diff[i] = candles[i]['price_close'] - candles[i - 1]['price_close']
    return p_diff


def load_data(data_path):
    if not os.path.exists(data_path):
        ValueError('Path {} does not exist.'.format(data_path))
    data = json.load(open(data_path, 'r'))
    print('--- Finished loading data at {} into memory ---'.format(data_path))
    return data


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
    if trade_stats['gross_return'] > 1.002:
        chart_pos = Chart(**chart_params, dirname=os.path.join(chart_pos_dir, '{}'.format(trade_cnt)))
        chart_pos.save()
    else:
        chart_neg = Chart(**chart_params, dirname=os.path.join(chart_neg_dir, '{}'.format(trade_cnt)))
        chart_neg.save()