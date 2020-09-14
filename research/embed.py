import numpy as np
from research.volume import load_data
from research.scatter import down_sample
from strategies.trailstop.backtest import is_breakout
from strategies.trailstop.backtest import calc_price_enter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import os
import click


# http://faculty.baruch.cuny.edu/smanzan/FINMETRICS/_book/time-series-models.html#trends-in-time-series


def trade_sampler(data, trade_params):
    fee = 0.001
    lb_breakout = trade_params['lb']
    lookback = trade_params['lookback']
    stop_coeff_initial = trade_params['stop_coeff_initial']
    stop_coeff = trade_params['stop_coeff']
    target_coeff = trade_params['target_coeff']
    terminal_num_periods = trade_params['terminal_num_periods']
    i = lookback

    while True:
        if i > len(data) - 1:
            break
        segment = data[i - lookback: i + 1]
        if is_breakout(segment, lb_breakout):
            stop_levels = [None] * len(segment)
            price_enter = calc_price_enter(segment, lb_breakout)  # data[i]['price_close']
            current_high = data[i]['price_high']
            stop_level = stop_coeff_initial * current_high
            target_level = target_coeff * price_enter
            hit_time_limit = False
            hit_stop = False
            time_since_entry = 1
            signal_period = i
            start_period = i - lookback
            i += 1
            candles = []
            for j in range(i - lookback, i + 1):
                candle = data[j]
                candle['period'] = j
                candles.append(candle)

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

                stop_level = current_high * stop_coeff
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


class GBMEmbedder(object):
    def __init__(self):
        pass

    def moments(self, x):
        # x shape (b, k)
        mu = np.mean(x, axis=1, keepdims=True)  # (b, 1)
        sigma = np.sqrt(np.std(x, axis=1, keepdims=True))  # (b, 1)
        return mu, sigma

    def embed(self, r):
        mu, sigma = self.moments(r)
        x = np.concatenate((mu, sigma), axis=1)
        return x


class GBMSplitEmbedder(GBMEmbedder):
    def __init__(self):
        super(GBMSplitEmbedder, self).__init__()

    def embed(self, r):
        if r.shape[1] < 2:
            mu1, sigma1 = self.moments(r)
            mu2, sigma2 = mu1, sigma1
        else:
            n = r.shape[1] // 2
            r1 = r[:, :n]
            r2 = r[:, n:]
            mu1, sigma1 = self.moments(r1)
            mu2, sigma2 = self.moments(r2)
        x = np.concatenate((mu1, sigma1, mu2, sigma2), axis=1)
        return x


class Experiment(object):
    def __init__(self, data_path, save_dir, trade_params, lb_return, embedder_type='gbm',
                 random_state=0, max_depth=4, min_samples_leaf=60):
        self.data_path = data_path

        self.agg_save_path = os.path.join(save_dir, 'agg_stats.md')
        self.group_save_path = os.path.join(save_dir, 'group_stats.md')
        self.tree_plot_save_path = os.path.join(save_dir, 'tree_plot.png')
        self.regions_plot_save_path = os.path.join(save_dir, 'regions_plot.png')

        self.trade_params = trade_params
        self.lb_return = lb_return
        self.embedder_type = embedder_type
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trades = self.get_trades(self.data_path)
        # trades_btc = self.get_trades('data/binance_spot_btc_usdt_1min.json')
        # self.trades = self.trades + trades_btc
        self.embedder = self.init_embedder()
        self.feature_names = self.init_feature_names()

    def run(self):
        X, y = self.get_plot_data()
        Xd, yd = down_sample(X, y)
        self.model = DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth,
                                            min_samples_leaf=self.min_samples_leaf)
        self.model.fit(Xd, yd)
        self.create_agg_stats_table(X, y)
        self.create_group_stats_table(X, y)
        self.create_plots(Xd, yd)

    def create_agg_stats_table(self, X, y):
        print('--- Aggregate Performance ---')
        y_pred = self.model.predict(X)
        m = confusion_matrix(y, y_pred)

        num_obs = len(y)
        num_pos = sum(m[1, :])
        num_neg = sum(m[0, :])
        num_pred_pos = sum(m[:, 1])
        num_pred_neg = sum(m[:, 0])
        num_true_pos_and_pred_pos = m[1, 1]
        num_true_neg_and_pred_neg = m[0, 0]

        table = [{
            'prob_true_given_pred_pos': num_true_pos_and_pred_pos / num_pred_pos,
            'prob_neg_given_pred_neg': num_true_neg_and_pred_neg / num_pred_neg,
            'prob_pos': num_pos / num_obs,
            'prob_neg': num_neg / num_obs,
            'num_obs': num_obs,
            'num_pos': num_pos,
            'num_neg': num_neg,
            'num_pred_pos': num_pred_pos,
            'num_pred_neg': num_pred_neg,
            'num_true_pos_and_pred_pos': num_true_pos_and_pred_pos,
            'num_true_neg_and_pred_neg': num_true_neg_and_pred_neg,
        }]

        df_table = pd.DataFrame(table)
        df_table.to_markdown(open(self.agg_save_path, 'w+'), index=False)
        print(df_table)

    def create_group_stats_table(self, X, y):
        print('--- Group Performance ---')
        table = []
        groups = self.model.apply(X)
        group_stats = {}

        for i in range(len(X)):
            g = groups[i]
            if g not in group_stats:
                group_stats[g] = {'num_obs': 0, 'num_pos': 0, 'prob_pos': 0.0, 'sum_return': 0.0,
                                  'avg_return_per_trade': 0.0}
            group_stats[g]['num_obs'] += 1
            if y[i] == 1:
                group_stats[g]['num_pos'] += 1
            group_stats[g]['sum_return'] += self.trades[i]['gross_return']
            group_stats[g]['prob_pos'] = group_stats[g]['num_pos'] / group_stats[g]['num_obs']
            group_stats[g]['avg_return_per_trade'] = group_stats[g]['sum_return'] / group_stats[g]['num_obs']

        for g, stats in group_stats.items():
            table.append({'group': g, 'prob_pos': stats['prob_pos'], 'num_pos': stats['num_pos'],
                          'num_obs': stats['num_obs'], 'avg_return_per_trade': group_stats[g]['avg_return_per_trade']})
        df_table = pd.DataFrame(table)
        df_table.to_markdown(open(self.group_save_path, 'w+'), index=False)
        print(df_table)

    def create_plots(self, Xd, yd):
        if len(self.feature_names) <= 2:
            mins = np.min(Xd, axis=0)
            maxs = np.max(Xd, axis=0)
            plt.figure()
            plot_decision_regions(Xd, yd, clf=self.model, colors="red,blue")
            plt.xlim(mins[0], maxs[0])
            plt.ylim(mins[1], maxs[1])
            plt.savefig(self.regions_plot_save_path)
            plt.close()

        plt.figure(figsize=(25, 17.5))
        plot_tree(self.model, class_names=["Neg", "Pos"], feature_names=self.feature_names, filled=True, rounded=True,
                  fontsize=14)
        plt.savefig(self.tree_plot_save_path)
        plt.close()

    def init_embedder(self):
        if self.embedder_type == 'gbm_split':
            embedder = GBMSplitEmbedder()
        else:
            embedder = GBMEmbedder()
        return embedder

    def init_feature_names(self):
        if self.embedder_type == 'gbm_split':
            return ["Mu1", "Sigma1", "Mu2", "Sigma2"]
        else:
            return ["Mu1", "Sigma1"]

    def get_trades(self, data_path):
        data = load_data(data_path)
        trades = []
        for i, trade in enumerate(trade_sampler(data, self.trade_params)):
            if i % 10 == 0:
                print(i)
            trades.append(trade)
        return trades

    def get_plot_data(self):
        X = []
        y = []
        for trade in self.trades:
            closes = [candle['price_close'] for candle in trade['candles'] if candle['period'] < trade['signal_period']]
            closes = np.array(closes)
            r = closes[1:] / closes[:len(closes) - 1]
            r = r.reshape(1, -1)
            X.append(self.embedder.embed(r))
            y.append(1 if trade['gross_return'] > self.lb_return else 0)

        X = np.concatenate(X, axis=0)
        y = np.array(y)
        return X, y


@click.command()
@click.option('--symbol', type=str, default='binance_spot_eth_usdt_1min')
@click.option('--embedder_type', type=str, default='gbm')  # gbm_split
@click.option('--max_depth', type=int, default=4)
@click.option('--min_samples_leaf', type=int, default=60)
@click.option('--lb_return', type=float, default=1.002)
def main(symbol, embedder_type, max_depth, min_samples_leaf, lb_return):
    data_path = 'data/{}.json'.format(symbol)
    save_dir = 'research/results/embed_{}_{}'.format(symbol, embedder_type)
    os.makedirs(save_dir, exist_ok=True)

    random_state = 0
    strategy_params = {
        'data_path': data_path,
        'lb': 1.005,
        'stop_coeff_initial': 0.985,
        'stop_coeff': 0.99,
        'target_coeff': 1.15,
        'terminal_num_periods': 20,
        'lookback': 60
    }
    e = Experiment(data_path, save_dir, strategy_params, lb_return, embedder_type, random_state, max_depth,
                   min_samples_leaf)
    e.run()


if __name__ == "__main__":
    main()



