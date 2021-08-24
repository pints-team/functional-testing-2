import altair as alt
import datetime
import numpy
import os
import pandas
import pathlib
import time

import pints.cptests


def run_the_test(method, problem, num_runs):
    """
    Run the given test (``problem``) and save the results into CSV files for
    now. Later, a cloud database.
    """
    # Base seed: then add run number for individual seeds
    base_seed = int(time.time())

    # Will be run from a push on the PINTS GH repo, so this is the PINTS git sha
    pints_sha = os.getenv('GITHUB_SHA') or f"unknown-{base_seed}"

    # The date and time that the tests were run
    date_time = datetime.datetime.now().replace(microsecond=0).isoformat()

    data_file = pathlib.Path('data') / method.__name__ / f'{problem.__name__}.csv'
    appending_to_file = data_file.is_file()
    if not appending_to_file:
        os.makedirs(data_file.parent, exist_ok=True)

    results = []
    for run in range(num_runs):
        seed = base_seed + run
        numpy.random.seed(seed)
        res = problem()
        res['pints_sha'] = pints_sha
        res['date_time'] = date_time
        res['seed'] = seed
        results.append(res)

    if appending_to_file:
        df = pandas.read_csv(data_file)
        for res in results[0].keys():
            assert res in df.columns, f'expected col in {data_file} called {res}'
        for col in df.columns:
            assert col in results[0].keys(), f'expected key in results dict called {col}'
    else:
        df = pandas.DataFrame()

    for res in results:
        df = df.append(res, ignore_index=True)

    with open(data_file, 'w') as f:
        f.write(df.to_csv(index=False))


def format_sha(sha_column):
    formatted_sha = []
    for sha in sha_column:
        if sha.startswith('unknown-'):
            formatted_sha.append(sha.replace('unknown-', ''))
        else:
            formatted_sha.append(sha[0:8])
    return formatted_sha


def plot_graph(method, test, df, col):
    chart = alt.Chart(df[["pints_sha", "sha_name", col, "date_time"]]).mark_point().encode(
        x=alt.X(
            field='sha_name',
            type='ordinal',
            sort={"field": "date_time"},
        ),
        y=alt.Y(
            field=col,
            type='quantitative',
            title=col.upper().replace('_', ' '),
        ),
        color=alt.Color('pints_sha', legend=None),
    ).properties(
        width=800,
        height=150
    ).interactive()

    output_dir = pathlib.Path('hugo_site') / 'static' / 'json' / method
    os.makedirs(output_dir, exist_ok=True)

    with open(output_dir / f'{problem}_{col}.json', 'w') as f:
        print(output_dir / f'{problem}_{col}.json')
        f.write(chart.to_json())


def plot_the_graphs(method, problem):
    """
    Plot the graphs for a given test, and dump the JSON out into JSON files in the hugo website
    directory
    """
    # TODO: Make .name() a classmethod so we can do this properly
    method, problem = method.__name__, problem.__name__

    data_file = pathlib.Path('data') / method / f'{problem}.csv'
    assert data_file.is_file(), f'{data_file} does not exist!'

    df = pandas.read_csv(data_file)
    df["sha_name"] = format_sha(df["pints_sha"])
    col_names = list(df.columns)

    if 'kld' in col_names:
        plot_graph(method, problem, df, 'kld')

    if 'distance' in col_names:
        plot_graph(method, problem, df, 'distance')

    if 'mean-ess' in col_names:
        plot_graph(method, problem, df, 'mean-ess')


if __name__ == "__main__":

    for method, problem in pints.cptests.tests():
        print(f'Testing {method.__name__} on {problem.__name__}')
        run_the_test(method, problem, num_runs=5)
        plot_the_graphs(method, problem)
