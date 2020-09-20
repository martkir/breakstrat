import click
import os
import json
import time
import sys
from strategies.common import run_backtest
from strategies.common import is_breakout
from strategies.common import calc_price_enter
from strategies.common import save_backtest_results


def trade_sampler(data, trade_params):
    lb_breakout = trade_params['lb_breakout']
    alpha = trade_params['alpha']
    beta = trade_params['beta']
    lookback = trade_params['lookback']
    stop_coeff_initial = trade_params['stop_coeff_initial']
    target_coeff = trade_params['target_coeff']
    terminal_num_periods = trade_params['terminal_num_periods']
    breakout_method = trade_params['breakout_method']
    breakout_params = {'alpha': alpha, 'beta': beta, 'lb_breakout': lb_breakout}

    i = lookback
    while True:
        if i > len(data) - 1:
            break
        segment = data[i - lookback: i + 1]

        if is_breakout(segment, breakout_params, breakout_method):
            stop_levels = [None] * len(segment)
            candles = segment
            price_enter = calc_price_enter(candles, breakout_params, breakout_method)
            current_high = data[i]['price_high']
            stop_level = stop_coeff_initial * price_enter

            target_level = target_coeff * price_enter
            hit_time_limit = False
            hit_stop = False
            time_since_entry = 1
            signal_period = i
            start_period = i - lookback
            i += 1

            while True:
                if i > len(data) - 1:
                    break

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
@click.option('--save_trades', is_flag=True)
@click.option('--no_print', is_flag=True)
@click.option('--data_path', type=str, default='data/binance_spot_eth_usdt_1min.json')
@click.option('--run_dir', type=str, default=None)
@click.option('--breakout_method', type=str, default='sdev')  # sdev
@click.option('--lb_breakout', type=float, default=1.0)
@click.option('--alpha', type=float, default=0.5)
@click.option('--beta', type=float, default=2.0)
@click.option('--stop_coeff_initial', type=float, default=0.985)
@click.option('--target_coeff', type=float, default=1.15)
@click.option('--terminal_num_periods', type=int, default=20)
@click.option('--lookback', type=int, default=60)
def main(with_plots, save_trades, no_print, data_path, run_dir, breakout_method, lb_breakout, alpha, beta,
         stop_coeff_initial, target_coeff, terminal_num_periods, lookback):
    if no_print:
        f = open(os.devnull, 'w')
        sys.stdout = f
    if not run_dir:
        run_dir = 'runs/baseline_{}'.format(time.time_ns() // 1000)

    trade_params = {
        'breakout_method': breakout_method,
        'lb_breakout': lb_breakout,
        'alpha': alpha,
        'beta': beta,
        'stop_coeff_initial': stop_coeff_initial,
        'target_coeff': target_coeff,
        'terminal_num_periods': terminal_num_periods,
        'lookback': lookback,
    }

    strategy_params = trade_params
    strategy_params['data_path'] = data_path
    os.makedirs(run_dir, exist_ok=True)
    json.dump(strategy_params, open(os.path.join(run_dir, 'params.json'), 'w+'), indent=4)
    data, trades, cum_returns, backtest_stats, summary_stats = \
        run_backtest(trade_sampler, data_path, trade_params, with_plots, run_dir)
    save_backtest_results(data, trades, cum_returns, backtest_stats, summary_stats, run_dir, save_trades)


if __name__ == '__main__':
    main()