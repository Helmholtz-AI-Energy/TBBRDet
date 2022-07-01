import requests
from pathlib import Path

ACCESS_TOKEN = 'insert your API token here'
params = {'access_token': ACCESS_TOKEN}

r = requests.get('https://zenodo.org/api/deposit/depositions', params={'access_token': ACCESS_TOKEN})
print(r.json())

bucket_url = r.json()[0]["links"]["bucket"]
print(bucket_url)

top_dir = Path('/path/to/dataset/archives/')

# This regex is specific to TBBR's naming
file_list = list((top_dir / 'train').glob('F*10[0-3]*.zst'))
print(list(file_list))

for f in list(file_list)[:1]:
    print(f'URL: {bucket_url}/{f.name}')
    ret = requests.put(f'{bucket_url}/{f.name}', data=open(f, 'rb'), params=params)
    print(ret.json())
