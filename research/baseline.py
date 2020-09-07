import click
import pandas as pd
import json
import os


@click.command()
@click.option('--run_ids', type=str, default="[1599403704228716, 1599480765138089]")
def main(run_ids):
    run_ids = json.loads(run_ids)
    run_dirs = ['runs/trailstop_{}'.format(run_id) for run_id in run_ids]
    results = []
    for run_dir in run_dirs:
        df_summary = pd.read_csv(os.path.join(run_dir, 'summary.csv'))
        srtategy_params = json.load(open(os.path.join(run_dir, 'params.json')))
        results.append({
            'run_dir': run_dir,
            'data_path': srtategy_params['data_path'],
            'num_trades': df_summary['num_trades'][0],
            'num_pos_trades': df_summary['num_pos_trades'][0],
            'num_neg_trades': df_summary['num_neg_trades'][0],
            'prob_true': df_summary['num_pos_trades'][0] / df_summary['num_trades'][0]
        })
    df_results = pd.DataFrame(results)
    os.makedirs('research/results', exist_ok=True)
    df_results.to_markdown(open('research/results/baseline.md', 'w+'), index=False)


if __name__ == '__main__':
    main()
