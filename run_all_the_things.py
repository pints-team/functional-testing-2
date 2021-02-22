import altair as alt
import datetime
import os
import pandas
import pathlib
import time

# Eventually we import loop over everything importable from pints.functionaltests that starts with
# Test*?
from pints.functionaltests import TestHaarioBardenetACMCOn2dimGaussianDistribution


def run_the_tests():
    """
    Run all the tests and shove the results into CSV files for now.  Later, a cloud database.
    """
    # Configurable, eventually?
    num_runs = 3

    # Will be run from a push on the PINTS GH repo, so this is the PINTS git sha
    pints_sha = os.getenv('GITHUB_SHA') or f"unknown-{int(time.time())}"

    # The date and time that the tests were run
    date_time = datetime.datetime.now().replace(microsecond=0).isoformat()

    data_file = pathlib.Path('data') / f'{TestHaarioBardenetACMCOn2dimGaussianDistribution.get_name()}.csv'
    assert data_file.is_file(), f'{data_file} does not exist!'

    results = [TestHaarioBardenetACMCOn2dimGaussianDistribution().get_results() for _ in range(num_runs)]

    df = pandas.read_csv(data_file)
    for res in results[0].keys():
        assert res in df.columns, f'expected col in {data_file} called {res}'

    for res in results:
        res['pints_sha'] = pints_sha
        res['date_time'] = date_time
        df = df.append(res, ignore_index=True)

    with open(data_file, 'w') as f:
        f.write(df.to_csv(index=False))


def plot_the_graphs():
    """
    Plot the graphs, and dump the JSON out into JSON files in the hugo website directory
    """

    test_name = TestHaarioBardenetACMCOn2dimGaussianDistribution.get_name()

    data_file = pathlib.Path('data') / f'{test_name}.csv'
    assert data_file.is_file(), f'{data_file} does not exist!'

    df = pandas.read_csv(data_file)

    chart_kld = alt.Chart(df).mark_point().encode(
        x='pints_sha',
        y='kld'
    )

    with open(pathlib.Path('hugo_site') / 'static_temp' / f'{test_name}_kld.json', 'w') as f:
        f.write(chart_kld.to_json())

    chart_ess = alt.Chart(df).mark_point().encode(
        x='pints_sha',
        y='mean-ess'
    )

    with open(pathlib.Path('hugo_site') / 'static_temp' / f'{test_name}_mean-ess.json', 'w') as f:
        f.write(chart_ess.to_json())


if __name__ == "__main__":
    run_the_tests()
    plot_the_graphs()
