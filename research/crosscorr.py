import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["figure.figsize"] = (16, 8)


def load_data(data_path):
    if not os.path.exists(data_path):
        ValueError('Path {} does not exist.'.format(data_path))
    data = json.load(open(data_path, 'r'))
    print('--- Finished loading data at {} into memory ---'.format(data_path))
    return data


def is_breakout(hits, lb=1.005):
    prices = [hit['price_close'] for hit in hits]  # hits[0] is oldest.
    prices = np.array(prices)
    returns = prices[1:] / prices[:len(prices) - 1]
    max_abs_past_return = np.max(np.abs(returns[:len(returns) - 1] - 1))
    last_return = returns[-1] - 1
    if last_return / max_abs_past_return > lb:
        return True
    else:
        return False


def trade_sampler(data, lb, lookback):
    stop_coeff_initial = 0.985
    stop_coeff = 0.99
    target_coeff = 1.05 * 4
    terminal_num_periods = 20
    i = lookback

    while True:
        if i > len(data) - 1:
            break
        segment = data[i - lookback: i + 1]

        if is_breakout(segment, lb):
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


def get_probs(eth_data, btc_data, lb, lookback):
    signal_periods_eth = []
    num_true_signals_eth = 0
    true_signal_periods_eth = set()
    for i, output in enumerate(trade_sampler(eth_data, lb, lookback)):
        signal_periods_eth.append(output['signal_period'])
        if output['gross_return'] > 1.005:
            num_true_signals_eth += 1
            true_signal_periods_eth.add(output['signal_period'])

    common_signal_periods = []
    conflict_signal_periods = []
    num_true_common_signals = 0
    num_true_conflict_signals = 0
    for i in signal_periods_eth:
        segment = btc_data[i - lookback: i + 1]
        if is_breakout(segment, lb=1.005):
            common_signal_periods.append(i)
            if i in true_signal_periods_eth:
                num_true_common_signals += 1
        else:
            conflict_signal_periods.append(i)
            if i in true_signal_periods_eth:
                num_true_conflict_signals += 1

    output = {
        'lb': lb,
        'prob_true': num_true_signals_eth / len(signal_periods_eth),
        'prob_true_common': num_true_common_signals / len(common_signal_periods),
        'num_true_signals': num_true_signals_eth,
        'num_signals': len(signal_periods_eth),
        'num_true_common_signals': num_true_common_signals,
        'num_common_signals': len(common_signal_periods)
    }
    return output


def main():
    lookback = 60
    save_dir = 'research/results/crosscorr_eth_usdt_1min'
    os.makedirs(save_dir, exist_ok=True)
    data_path_target = 'data/binance_spot_eth_usdt_1min.json'
    data_path_aux = 'data/binance_spot_btc_usdt_1min.json'

    data_main = load_data(data_path=data_path_target)
    data_aux = load_data(data_path=data_path_aux)
    table = []

    lowerbounds = [1.0, 1.0025, 1.005, 1.01]
    # lowerbounds = [1.09]  # todo: treshold is way to high??

    with tqdm(total=len(lowerbounds)) as pbar:
        for lb in lowerbounds:
            record = get_probs(data_main, data_aux, lb, lookback)
            record['data_path_target'] = data_path_target
            record['data_path_aux'] = data_path_aux
            table.append(get_probs(data_main, data_aux, lb, lookback))
            description = 'lb: {}'.format(lb)
            pbar.set_description(description)
            pbar.update(1)

    prob_true_list = [table[i]['prob_true'] for i in range(len(table))]
    prob_true_common_list = [table[i]['prob_true_common'] for i in range(len(table))]

    plt.figure()
    plt.plot(lowerbounds, prob_true_list, marker='.', linewidth=0.5, color='blue', label='single')
    plt.plot(lowerbounds, prob_true_common_list, marker='.', linewidth=0.5, color='magenta', label='common')
    plt.xticks(lowerbounds)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'plot.png'))
    plt.close()

    df_table = pd.DataFrame(table)
    df_table.to_markdown(open(os.path.join(save_dir, 'table.md'), 'w+'), index=False)
    print('--- Finished saving table {} ---'.format(os.path.join(save_dir, 'table.md')))
    print(df_table)

if __name__ == '__main__':
    main()
