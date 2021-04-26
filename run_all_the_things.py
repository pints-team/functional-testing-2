import altair as alt
import datetime
import numpy
import os
import pandas
import pathlib
import time

import pints.functionaltests


def get_list_of_tests():
    """
    Inspect the pints.functionaltests module for all methods beginning with 'test_'
    """
    return [func for func in dir(pints.functionaltests) if func.startswith('test_')]


def get_method_and_problem_from_test_name(test_name):
    """
    Turn a test name into a tuple of method and problem names, e.g.
    test_haario_bardenet_acmc_on_two_dim_gaussian -> (haario_bardenet_acmc, two_dim_gaussian)
    """
    return test_name.replace('test_', '', 1).split(sep='_on_')


def run_the_test(test_name, num_runs):
    """
    Run the given test and save the results into CSV files for now.  Later, a cloud database.
    """
    # Base seed: then add run number for individual seeds
    base_seed = int(time.time())

    # Will be run from a push on the PINTS GH repo, so this is the PINTS git sha
    pints_sha = os.getenv('GITHUB_SHA') or f"unknown-{base_seed}"

    # The date and time that the tests were run
    date_time = datetime.datetime.now().replace(microsecond=0).isoformat()

    method, problem = get_method_and_problem_from_test_name(test_name)

    data_file = pathlib.Path('data') / method / f'{problem}.csv'
    assert data_file.is_file(), f'{data_file} does not exist!'

    results = []
    for run in range(num_runs):
        seed = base_seed + run
        numpy.random.seed(seed)
        res = getattr(pints.functionaltests, test)()
        res['pints_sha'] = pints_sha
        res['date_time'] = date_time
        res['seed'] = seed
        results.append(res)

    df = pandas.read_csv(data_file)
    for res in results[0].keys():
        assert res in df.columns, f'expected col in {data_file} called {res}'
    for col in df.columns:
        assert col in results[0].keys(), f'expected key in results dict called {col}'

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


def plot_the_graphs(test_name):
    """
    Plot the graphs for a given test, and dump the JSON out into JSON files in the hugo website
    directory
    """
    method, problem = get_method_and_problem_from_test_name(test_name)

    data_file = pathlib.Path('data') / method / f'{problem}.csv'
    assert data_file.is_file(), f'{data_file} does not exist!'

    df = pandas.read_csv(data_file)
    df["sha_name"] = format_sha(df["pints_sha"])

    chart_kld = alt.Chart(df[["pints_sha", "sha_name", "kld", "date_time"]]).mark_point().encode(
        x=alt.X(
            field='sha_name',
            type='ordinal',
            sort={"field": "date_time"},
        ),
        y=alt.Y(
            field='kld',
            type='quantitative',
            title='KLD',
        ),
        color=alt.Color('pints_sha', legend=None),
    ).properties(
        width=800,
        height=150
    ).interactive()

    with open(pathlib.Path('hugo_site') / 'static' / 'json' / method / f'{problem}_kld.json', 'w') as f:
        f.write(chart_kld.to_json())

    chart_ess = alt.Chart(df[["pints_sha", "sha_name", "mean-ess", "date_time"]]).mark_point().encode(
        x=alt.X(
            field='sha_name',
            type='ordinal',
            sort={"field": "date_time"},
            title="commit sha",
        ),
        y=alt.Y(
            field='mean-ess',
            type='quantitative',
            title='mean ESS',
        ),
        color=alt.Color('pints_sha', legend=None),
    ).properties(
        width=800,
        height=150
    ).interactive()

    with open(pathlib.Path('hugo_site') / 'static' / 'json' / method / f'{problem}_mean-ess.json', 'w') as f:
        f.write(chart_ess.to_json())


if __name__ == "__main__":

    for test in get_list_of_tests():
        run_the_test(test_name=test, num_runs=5)
        plot_the_graphs(test)
