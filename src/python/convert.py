'''
Convert Yelp Academic Dataset from JSON to CSV

Requires Pandas (https://pypi.python.org/pypi/pandas)

By Paul Butler, No Rights Reserved
'''

import json
import pandas as pd
from glob import glob
import sys

def convert(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob


def jsonToCsv(json_filenames):
    for json_filename in glob(json_filenames):
        csv_filename = '%s.csv' %(json_filename[:-5])
        print 'Converting %s to %s' % (json_filename, csv_filename)
        df = pd.DataFrame([convert(line) for line in file(json_filename)])
        df.to_csv(csv_filename, encoding='utf-8', index=False)

        


if  __name__ == '__main__':
    jsonToCsv(sys.argv[1])


