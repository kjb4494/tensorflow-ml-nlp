import pandas as pd
import json
import numpy as np


def get_data_frame(input_file_path):
    record_path = ['data', 'paragraphs', 'qas', 'answers']
    print('Reading the json file...')
    file = json.loads(open(input_file_path).read())
    print('processing...')
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    result = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1, sort=False).reset_index()
    result['c_id'] = result['context'].factorize()[0]
    print('Done. shape: {}'.format(result.shape))
    return result
