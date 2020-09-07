import click
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.figsize"] = (16, 8)


def get_raw_volume(data_path, time_periods):
    data = load_data(data_path)
    x = []
    for i in time_periods:
        x.append(data[i]['volume_traded'])
    return x


def get_targets(backtest_stats):
    y = []
    for i in range(len(backtest_stats)):
        if backtest_stats[i]['gross_return'] > 1.0:
            y.append(1)
        else:
            y.append(0)
    return y


def make_histogram(x, targets, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    colors = []
    x_pos = []
    x_neg = []
    for i in range(len(x)):
        if targets[i] == 1:
            x_pos.append(x[i])
            colors.append('blue')
        else:
            x_neg.append(x[i])
            colors.append('red')
    plt.figure()
    plt.hist(x_pos, bins=100, alpha=0.3, density=True, color='blue')
    plt.hist(x_neg, bins=100, alpha=0.3, density=True, color='red')
    save_path = os.path.join(save_dir, 'hist.png')
    plt.savefig(save_path)
    plt.close()
    print('--- Finished saving histogram at {} ---'.format(save_path))


def make_prob_table_and_plot(x, targets, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sort_idxs = np.argsort(x)
    x = np.sort(x)
    targets = [targets[i] for i in sort_idxs]
    x_segments = np.array_split(x, 10)
    target_segments = np.array_split(targets, 10)
    table = []
    probs = []
    for xseg, yseg in zip(x_segments, target_segments):
        prob = sum(yseg) / len(yseg)
        table.append({
            'interval': '[{}, {}]'.format(xseg[0], xseg[-1]),
            'prob_true': prob,
            'num_obs': len(yseg)
        })
        probs.append(prob)
    df_table = pd.DataFrame(table)
    df_table.to_markdown(open(os.path.join(save_dir, 'prob_table.md'), 'w+'), index=False)

    plt.figure()
    plt.ylim(0, 1)
    plt.plot([i for i in range(len(probs))], probs, linewidth=0.5, marker='.', color='blue')
    save_path = os.path.join(save_dir, 'prob_plot.png')
    plt.savefig(save_path)
    plt.close()


def load_data(data_path):
    if not os.path.exists(data_path):
        ValueError('Path {} does not exist.'.format(data_path))
    data = json.load(open(data_path, 'r'))
    print('--- Finished loading data at {} into memory ---'.format(data_path))
    return data


@click.command()
@click.option('--run_dir', type=str, default="runs/trailstop_1599403704228716")  # runs/trailstop_1599480765138089
# @click.option('--run_dir', type=str, default="runs/trailstop_1599480765138089")
@click.option('--save_dir', type=str, default="research/results/volume_eth_usdt_5min")
def main(run_dir, save_dir):
    strategy_params = json.load(open(os.path.join(run_dir, 'params.json')))
    df_backtest_stats = pd.read_csv(os.path.join(run_dir, 'trades.csv'))
    backtest_stats = df_backtest_stats.to_dict('records')
    enter_periods = [backtest_stats[i]['enter_period'] for i in range(len(backtest_stats))]
    x = get_raw_volume(strategy_params['data_path'], enter_periods)
    targets = get_targets(backtest_stats)
    make_histogram(x, targets, save_dir)
    make_prob_table_and_plot(x, targets, save_dir)


if __name__ == '__main__':
    main()
