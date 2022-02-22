# Toshiaki Koike-Akino, 2022
# Data loading: https://github.com/datasets/exchange-rates
import pandas as pd
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# args example
def get_args(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser('forex loader')
    parser.add_argument('--verb', action='store_true')
    parser.set_defaults(verb=True)
    # loader args
    add_args(parser)
    return parser.parse_args()

# loader-specific args
def add_args(parser):
    # data loader args
    parser = parser.add_argument_group('forex')
    parser.add_argument('--path', default='exchange-rates/data', help='data path')
    parser.add_argument('--data', default='annual', choices=['daily', 'monthly', 'annual'], help='data freq.')
    parser.add_argument('--country', default=[], nargs='*', type=str, 
                        #choices=['Australia' 'Austria' 'Belgium' 'Brazil' 'Canada' 'China' 'Denmark'
                        #         'Euro' 'Finland' 'France' 'Germany' 'Greece' 'Hong Kong' 'India'
                        #         'Ireland' 'Italy' 'Japan' 'Malaysia' 'Mexico' 'Netherlands' 'New Zealand'
                        #         'Norway' 'Portugal' 'Singapore' 'South Africa' 'South Korea' 'Spain'
                        #         'Sri Lanka' 'Sweden' 'Switzerland' 'Taiwan' 'Thailand' 'United Kingdom'
                        #         'Venezuela'],
                        help='country list')
    parser.add_argument('--dates', default=['1971', '2019'], nargs=2, help='dates range')

# load exchange-rates/data/?.csv
def load_data(args, verb=False):
    fdata = os.path.join(args.path, f'{args.data}.csv')
    if verb: print('# loading', fdata)
    df = pd.read_csv(fdata)
    return df

# plot dataframe
def plot(df, title='forex', show=False, verb=False):
    plt.figure()
    #plt.plot(df['Date'], df['Exchange rate'])
    df.plot(x='Date', logy=True)
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
def reframe(args, df, country=None, verb=False, savefig=False):
    # re-frame data for date vs. countries
    data = pd.DataFrame({'Date': []}) # empty
    
    #if verb: print(data.shape)
    for c in country:
        # per-country data
        d = df[df['Country'] == c]
        if verb: print(c, d.shape, np.min(d['Date']), np.max(d['Date']))

        # re-label 'Exchange rate' to country name
        d = d.drop(columns={'Country'})
        d = d.rename(columns={'Exchange rate': c})
        #print(d.head())

        # merge country date
        data = d if data.size == 0 else pd.merge(data, d, on='Date', how='inner') 
        #print(data.shape)

        # plot data
        if savefig:
            plot(d, title=f'{c}_{args.data}', verb=verb)
    return data

# select country
def get_country(args):
    #country = np.unique(df['Country'].to_numpy())
    #country = ['Australia', 'Euro', 'Ireland', 'New Zealand', 'United Kingdom']
    '''
    # available
    country = ['Australia' 'Austria' 'Belgium' 'Brazil' 'Canada' 'China' 'Denmark'
               'Euro' 'Finland' 'France' 'Germany' 'Greece' 'Hong Kong' 'India'
               'Ireland' 'Italy' 'Japan' 'Malaysia' 'Mexico' 'Netherlands' 'New Zealand'
               'Norway' 'Portugal' 'Singapore' 'South Africa' 'South Korea' 'Spain'
               'Sri Lanka' 'Sweden' 'Switzerland' 'Taiwan' 'Thailand' 'United Kingdom'
               'Venezuela']
    '''

    # args.country if given
    if len(args.country) > 0:
        return args.country

    # if empty, depending on data
    if args.data == 'annual': # countries having data 47years for 1971-01-01:2017-01-01
        country = ['Australia', 'Canada', 'Denmark', 'Japan', 'Malaysia', 'New Zealand',
                   'Norway', 'South Africa', 'Sweden', 'Switzerland']
    elif args.data == 'monthly': # countries having data 574 months for 1971-01-01:2018-10-01
        country = ['Australia', 'Canada', 'Denmark', 'Japan', 'Malaysia', 'New Zealand',
                   'Norway', 'South Africa', 'Sweden', 'Switzerland', 'United Kingdom']
    elif args.data == 'daily': # countries having data 12465 days for 1971-01-04:2018-10-12
        country = ['Australia', 'Canada', 'Denmark', 'Japan', 'Malaysia', 'New Zealand',
                   'Norway', 'Sweden', 'Switzerland', 'United Kingdom']
    else:
        raise NameError(args.country, args.data)
    
    return country

# main script
def main(args, drop_date=False, verb=False):
    # load data
    df = load_data(args, verb=verb)
    if verb:
        print('# loaded data', args.data) 
        print(df.head())
        print(df.shape)

    # select countries
    country = get_country(args)

    # reframe dataset: date vs. countries
    df = reframe(args, df, country, verb=verb, savefig=False)
    if verb: 
        print('# selected country', country)
        print(df.head())
        print(df.shape)

    # select dates range
    df = select_dates(df, args.dates)
    if verb: 
        print('# selected dates', args.dates)
        print(df.head())
        print(df.shape)

    # drop date
    if drop_date:
        df = df.drop(columns=['Date'])
    return df

# select dates range
def select_dates(df, dates=['1971', '2019']):
    mask = (df['Date'] >= dates[0]) & (df['Date'] <= dates[1])
    return df.loc[mask]

# example use
if __name__ == '__main__':
    # example args
    args = get_args()
    if args.verb: print('#', args)

    # load data
    df = main(args, verb=args.verb)

    # plot data
    plot(df, title=f'forex_{args.data}', verb=args.verb)
