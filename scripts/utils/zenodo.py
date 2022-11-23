import requests
from pathlib import Path

'''
Simple tool to upload files to a Zenodo record.
This has not been extensively developed so expect to need
to make a couple of tweaks for your particular setup.

NOTE: If you have created the record for the first time on
the website, you need to add at least one file for it to create
a bucket link.
We recommend uploading and deleting a dummy file without publishing
the record.
'''

DEPOSIT = 'insert deposit number (number in URL of the record)'
ACCESS_TOKEN = 'insert your API token here'
params = {'access_token': ACCESS_TOKEN}

r = requests.get(f'https://zenodo.org/api/deposit/depositions/{DEPOSIT}', params={'access_token': ACCESS_TOKEN})
print(r.json())

bucket_url = r.json()["links"]["bucket"]
print(bucket_url)

# Insert you directory containing files to upload here
top_dir = Path('/path/to/dataset/archives/')

# Here you might have a different archive format
file_list = list(top_dir.glob('*.zst'))
print(list(file_list))

for f in list(file_list):
    print(f'URL: {bucket_url}/{f.name}')
    ret = requests.put(f'{bucket_url}/{f.name}', data=open(f, 'rb'), params=params)
    print(ret.json())
