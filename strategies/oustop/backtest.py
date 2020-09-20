import click
import os
import json
import numpy as np
import time
import sys
from strategies.utils import run_backtest
from strategies.utils import is_breakout
from strategies.utils import calc_price_enter
from strategies.utils import save_backtest_results


def ou_param_predict(t, s0, mu, sigma, theta):
    # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    popmean = s0 * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))
    popvar = (1 / 2 * theta) * np.square(sigma) * (1 - np.exp(-2 * theta * t))
    return popmean, popvar


def ou_price_predict(t, s0, mu, sigma, theta):
    # https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    # print('t: {} s0: {} mu: {} sigma: {} theta: {}'.format(t, s0, mu, sigma, theta))
    # popmean = s0 * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))
    mult = np.exp(-theta * t)  # between (0, 1] theta large -> 0
    popmean = (s0 - mu) * mult + mu
    # print('popmean 1: ', popmean, 'diff: ', s0 - mu, 'mu: ', mu, 'exp: ', np.exp(-theta * t))
    popvar = (1 / 2 * theta) * np.square(sigma) * (1 - np.exp(-2 * theta * t))
    return popmean, popvar


def est_ou_sigma(diffs, method, lag):
    start_idx = max([1, len(diffs) - lag])
    if method == 'max_abs_diff':
        x = np.max(np.abs(diffs[start_idx:]))
        sigma = 0.9 * x
    elif method == 'sdev_abs_diff':
        x = diffs[start_idx:]
        sigma = np.std(x)
    else:
        raise ValueError('Method {} is invalid.'.format(method))
    return sigma


def get_ou_stop_level(candles, signal_period, signal_low, last_peak, time_since_peak, est_method, est_lag, theta,
                      pred_method):
    # theta: the larger theta the more close offset is from peak (less sensitive to recent price)
    c = 1
    diffs = np.array([None] * len(candles))
    for i in range(1, len(candles)):
        diffs[i] = c * candles[i]['price_close'] - candles[i - 1]['price_close']
    if candles[-1]['period'] == signal_period:
        last_peak = candles[-1]['price_high']
        time_since_peak = 0
    if candles[-1]['period'] > signal_period:
        if candles[-1]['price_high'] > last_peak:
            last_peak = candles[-1]['price_high']
            time_since_peak = 0
    mu = last_peak
    sigma = est_ou_sigma(diffs, est_method, est_lag)
    if pred_method == 'start_at_peak':
        s0 = last_peak
        price_pred, price_var = ou_price_predict(time_since_peak + 1, s0, mu, sigma, theta)
    elif pred_method == 'start_at_curr':
        s0 = candles[-1]['price_low']
        price_pred, price_var = ou_price_predict(1, s0, mu, sigma, theta)
    else:
        raise ValueError('Method {} is invalid.'.format(pred_method))
    # next_stop_level = max([price_pred - 3.0 * np.sqrt(price_var), signal_low])
    next_stop_level = price_pred - 3.0 * np.sqrt(price_var)
    return next_stop_level, last_peak, time_since_peak + 1


def trade_sampler(data, trade_params):
    fee = 0.001
    lb_breakout = trade_params['lb_breakout']
    alpha = trade_params['alpha']
    beta = trade_params['beta']
    lookback = trade_params['lookback']
    target_coeff = trade_params['target_coeff']
    terminal_num_periods = trade_params['terminal_num_periods']
    est_method = trade_params['est_method']
    est_sigma_lag = trade_params['est_sigma_lag']
    theta = trade_params['theta']
    pred_method = trade_params['pred_method']
    breakout_method = trade_params['breakout_method']
    breakout_params = {'alpha': alpha, 'beta': beta, 'lb_breakout': lb_breakout}

    i = lookback
    while True:
        if i > len(data) - 1:
            break
        segment = data[i - lookback: i + 1]
        if is_breakout(segment, breakout_params, breakout_method):
            last_peak = None
            stop_levels = [None] * len(segment)
            price_enter = calc_price_enter(segment, breakout_params, breakout_method)  # data[i]['price_close']

            current_high = data[i]['price_high']
            target_level = target_coeff * price_enter
            hit_time_limit = False
            hit_stop = False
            time_since_entry = 1
            signal_period = i
            signal_low = data[i]['price_low']
            start_period = i - lookback
            candles = []
            for j in range(i - lookback, i + 1):  # todo: should be i - lookback + 1
                candle = data[j]
                candle['period'] = j
                candles.append(candle)

            time_since_peak = 0
            stop_level, last_peak, time_since_peak = get_ou_stop_level(
                candles, signal_period, signal_low, last_peak, time_since_peak, est_method, est_sigma_lag, theta,
                pred_method)

            i += 1
            while True:
                if i > len(data) - 1:
                    break

                candle = data[i]
                candle['period'] = i
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

                stop_level, last_peak, time_since_peak = get_ou_stop_level(
                    candles, signal_period, signal_low, last_peak, time_since_peak, est_method, est_sigma_lag, theta,
                    pred_method)

                i += 1
                time_since_entry += 1

            output = {
                'candles': candles,
                'gross_return': gross_return,
                'net_return': gross_return * (1 - fee) ** 2,
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
@click.option('--save_trades', is_flag=True)
@click.option('--no_print', is_flag=True)
@click.option('--data_path', type=str, default='data/binance_spot_eth_usdt_5min.json')
@click.option('--run_dir', type=str, default=None)
@click.option('--lb_breakout', type=float, default=1.005)
@click.option('--alpha', type=float, default=0.5)
@click.option('--beta', type=float, default=2.0)
@click.option('--target_coeff', type=float, default=1.15 * 4)
@click.option('--terminal_num_periods', type=int, default=100)
@click.option('--lookback', type=int, default=60)
@click.option('--est_method', type=str, default='max_abs_diff')  # max_abs_diff, sdev_abs_diff
@click.option('--est_sigma_lag', type=int, default=100)
@click.option('--theta', type=float, default=2.0)
@click.option('--pred_method', type=str, default='start_at_peak')  # start_at_peak, start_at_curr
@click.option('--breakout_method', type=str, default='abs')  # abs, sdev
def main(with_plots, save_trades, no_print, data_path, run_dir, lb_breakout, alpha, beta, target_coeff,
         terminal_num_periods, lookback, est_method, est_sigma_lag, theta, pred_method, breakout_method):
    # theta can be only hparam -> means high variance (low reversion to mean).
    trade_params = {
        'lb_breakout': lb_breakout,
        'alpha': alpha,
        'beta': beta,
        'target_coeff': target_coeff,
        'terminal_num_periods': terminal_num_periods,
        'lookback': lookback,
        'est_method': est_method,
        'est_sigma_lag': est_sigma_lag,
        'theta': theta,
        'pred_method': pred_method,
        'breakout_method': breakout_method
    }
    if no_print:
        f = open(os.devnull, 'w')
        sys.stdout = f
    if not run_dir:
        run_dir = 'runs/oustop_{}'.format(time.time_ns() // 1000)
    strategy_params = trade_params
    strategy_params['data_path'] = data_path
    os.makedirs(run_dir, exist_ok=True)
    json.dump(strategy_params, open(os.path.join(run_dir, 'params.json'), 'w+'), indent=4)
    data, trades, cum_returns, backtest_stats, summary_stats = \
        run_backtest(trade_sampler, data_path, trade_params, with_plots, run_dir)
    save_backtest_results(data, trades, cum_returns, backtest_stats, summary_stats, run_dir, save_trades)


if __name__ == '__main__':
    main()
