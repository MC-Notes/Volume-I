import yaml, requests, sys, json

argv = sys.argv[1:]

try:
    url, access, metadata_file, notebook, requirements = argv
except:
    print('Usage: python zenodo_upload_doi.py zenodo_url zenodo_access_token metadata_file \n upload to zenodo url (has to have access token included) with metadata stored as .yml file in metadata_file. ')
    sys.exit(2)

headers = {"Content-Type": "application/json"}
with open(metadata_file) as f:
    metadata = yaml.load(f)

metadata.set_default('upload_type', 'publication')
metadata.set_default('publication_type', 'other')
metadata.set_default('access_right', 'open')

import time
l = time.localtime()
metadata.set_default('publication_date', "{tm_year}-{tm_mon:>02}-{tm_mday:>02}".format(tm_year=l.tm_year, tm_mon=l.tm_mon, tm_mday=l.tm_mday))

def check_metadata(metadata):
    if isinstance(metadata, str):
        return True
    elif isinstance(metadata, dict):
        return all(map(check_metadata, metadata.values()))
    elif isinstance(metadata, list):
        return all(map(check_metadata, metadata))
    raise AttributeError('Type mismatch in given metadata: {} of type {!s}'.format(metadata, type(metadata)))
    
c_url = '{}?access_token={}'.format(url, access)

r = requests.post(, data=json.dumps(dict(metadata=metadata)), headers=headers)
r_json = r.json()

if r_json['status'] != 201:
    stat = 'Given metadata returned error code {}'.format(r_json['status'])
    if 'message' in r_json:
        stat = "{} with message {}".format(stat, r_json['message'])
    raise AttributeError(stat)
    sys.exit(r_json['status'])
else:
    post_url = '{}?access_token={}'.format(r_json['links']['files'], access)

    with open(notebook, 'rb') as f:
        data = {'filename': 'notebook.ipynb'}
        files = {'file': f}
        r = requests.post(post_url, data=data, files=files)
        
    with open(requirements, 'rb') as f:
        data = {'filename': 'requirements.txt'}
        files = {'file': f}
        r = requests.post(post_url, data=data, files=files)
        
    pub_url = '{}?access_token={}'.format(r_json['links']['publish'], access)
    import pprint
    r = requests.post(pub_url)
    pprint.pprint(r.json)