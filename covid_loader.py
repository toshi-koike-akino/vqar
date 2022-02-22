# Toshiaki Koike-Akino, 2022
# Data loading: https://github.com/nytimes/covid-19-data
import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# args example
def get_args(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser('covid loader')
    parser.add_argument('--verb', action='store_true')
    parser.set_defaults(verb=True)
    # loader args
    add_args(parser)
    return parser.parse_args()

# loader-specific args
def add_args(parser):
    # data loader args
    parser = parser.add_argument_group('covid')
    parser.add_argument('--path', default='covid-19-data', help='data path')
    parser.add_argument('--data', default='us-states', 
                        choices=['us', 'us-states'], 
                        help='csv file.')
    parser.add_argument('--states', default=['Massachusetts', 'Maine', 'New Hampshire', 'New York', 'Connecticut'], 
                        nargs='*', type=str, 
                        help='states list')
    parser.add_argument('--periods', default=7, type=int, 
                        help='difference periods (days). 0: cumsum, 1: daily diff, 7: weekly diff')
    parser.add_argument('--dates', default=['2021', '2023'], nargs=2, help='dates range')
    parser.add_argument('--decimate', default=7, type=int, help='resampling decimation factor')

# load covid-19-data/?.csv
def load_data(args, verb=False):
    fdata = os.path.join(args.path, f'{args.data}.csv')
    if verb: print('# loading', fdata)
    df = pd.read_csv(fdata)
    return df

# plot dataframe
def plot(df, title='covid', show=False, verb=False):
    plt.figure()
    df.plot(x='date', logy=True)
    plt.title(title)
    plt.grid()

    path = 'plots'
    os.makedirs(path, exist_ok=True)
    fname = f'{title}.png'
    plt.savefig(os.path.join(path, fname))
    if verb: print('# saving', fname)
    if show:
        plt.show()

# re-format dataframe
def reframe_states(args, df, verb=False, savefig=False):
    # re-frame data for date vs. countries
    data = pd.DataFrame({'date': []}) # empty
    
    #if verb: print(data.shape)
    for target in args.states:
        # per-state data
        d = df[df['state'] == target]
        if verb: print(target, d.shape, np.min(d['date']), np.max(d['date']))

        # re-label 'Exchange rate' to country name
        d = d.drop(columns={'state', 'deaths', 'fips'})
        d = d.rename(columns={'cases': target})
        #print(d.head())

        # merge country date
        data = d if data.size == 0 else pd.merge(data, d, on='date', how='inner') 
        #print(data.shape)

        # plot data
        if savefig:
            plot(d, title=f'{c}_{args.data}', verb=verb)
    return data

# main script
def main(args, drop_date=False, verb=False):
    # load data
    df = load_data(args, verb=verb)
    if verb:
        print('# loaded data', args.data) 
        print(df.head())
        print(df.shape)

    # reframe dataset: date vs. countries
    if args.data == 'us-states':
        df = reframe_states(args, df, verb=verb, savefig=False)
        if verb: 
            print('# selected states', args.states)
            print(df.head())
            print(df.shape)

    # inverse cumsum
    df = diff(df, periods=args.periods)

    # select dates range
    df = select_dates(df, args.dates)
    if verb: 
        print('# selected dates', args.dates)
        print(df.head())
        print(df.shape)
        
    # resampling
    df = df[df.index % args.decimate == 0]
    if verb:
        print('# resampling', args.decimate)
        print(df.head())
        print(df.shape)

    # drop date
    if drop_date:
        df = df.drop(columns=['date'])
    return df

# inverse cumsum (diff): 0 do nothing
def diff(df, periods=0):
    if periods == 0:
        return df

    # exclude 'date' column
    dates = df['date']
    #print(dates.head())
    df = df.drop(columns=['date'])
    #print('dropped')
    #print(df.head())

    # diff
    df = df.diff(periods=periods)
    
    # revert back 'date' column
    df.insert(0, 'date', dates)

    # remove head NaN
    if periods > 0:
        df = df.drop(df.index[:periods])
    return df


# select dates range
def select_dates(df, dates=['2021', '2022']):
    mask = (df['date'] >= dates[0]) & (df['date'] <= dates[1])
    return df.loc[mask]

# example use
if __name__ == '__main__':
    # example args
    args = get_args()
    if args.verb: print('#', args)

    # load data
    df = main(args, verb=args.verb)

    # plot data
    plot(df, title=f'covid_{args.data}', verb=args.verb)
