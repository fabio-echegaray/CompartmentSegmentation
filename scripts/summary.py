import os
import argparse
import javabridge

import pandas as pd

from cached import CachedImageFile
from logger import get_logger

log = get_logger(name='summary')


def process_dir(path) -> pd.DataFrame:
    out = pd.DataFrame()
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            ext = filename.split('.')[-1]
            if ext == 'mvd2':
                try:
                    joinf = os.path.join(root, filename)
                    log.info(f'Processing {joinf}')
                    img_struc = CachedImageFile(joinf, cache_results=False)
                    out = out.append(img_struc.info, ignore_index=True)
                except FileNotFoundError as e:
                    log.warning(f'Data not found for file {joinf}.')

    return out


if __name__ == '__main__':
    description = 'Generate pandas dataframe summary of microscope images stored in the specified path (recursively).'
    epilogue = '''
    The outputs are two files in Excel and comma separated values (CSV) formats, i.e., summary.xlsx and summary.csv.
    '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='Path where to start the search.')
    args = parser.parse_args()

    df = process_dir(args.path)
    print(df)
    df.set_index(['filename', 'instrument_id', 'pixels_id']).to_excel('summary.xlsx', merge_cells=True)
    df.to_csv('summary.csv')

    javabridge.kill_vm()
