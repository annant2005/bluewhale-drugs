import urllib.request
import gzip
import shutil
import os

url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
out_gz = 'tox21.csv.gz'
out_csv = 'tox21.csv'

print('Downloading Tox21 dataset...')
urllib.request.urlretrieve(url, out_gz)
print('Extracting...')
with gzip.open(out_gz, 'rb') as f_in, open(out_csv, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
print('Done! tox21.csv is ready.')
os.remove(out_gz) 