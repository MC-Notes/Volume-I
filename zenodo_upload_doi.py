import yaml, requests, sys, json

argv = sys.argv[1:]

try:
    url, access_token, metadata_file, notebook, requirements = argv
except:
    print('Usage: python zenodo_upload_doi.py zenodo_url zenodo_access_token metadata_file \n upload to zenodo url (has to have access token included) with metadata stored as .yml file in metadata_file. ')
    sys.exit(2)

###############
# Make sure metadata constains necessary info
headers = {"Content-Type": "application/json"}
with open(metadata_file) as f:
    metadata = yaml.load(f)

metadata.set_default('upload_type', 'publication')
metadata.set_default('publication_type', 'other')
metadata.set_default('access_right', 'open')

import time
l = time.localtime()
metadata.set_default('publication_date', "{tm_year}-{tm_mon:>02}-{tm_mday:>02}".format(tm_year=l.tm_year, tm_mon=l.tm_mon, tm_mday=l.tm_mday))
###############

# simple check for metadata
def check_metadata(metadata):
    if isinstance(metadata, str):
        return True
    elif isinstance(metadata, dict):
        return all(map(check_metadata, metadata.values()))
    elif isinstance(metadata, list):
        return all(map(check_metadata, metadata))
    raise AttributeError('Type mismatch in given metadata: {} of type {!s}'.format(metadata, type(metadata)))
check_metadata(metadata)

# Make small script to add access token to urls:
def access(url):
    return '{}?access_token={}'.format(url, access)

# create deposition
c_url = access(url)
r = requests.post(c_url, data=json.dumps(dict(metadata=metadata)), headers=headers)
r_json = r.json()

# check deposition status
if r_json['status'] != 201:
    # deposition failed, report status
    stat = 'Given metadata returned error code {}'.format(r_json['status'])
    if 'message' in r_json:
        stat = "{} with message {}".format(stat, r_json['message'])
    raise AttributeError(stat)
    sys.exit(r_json['status'])
else:
    # deposition success:
    post_url = access(r_json['links']['files'])
    # add notebook and requirements to deposition
    with open(notebook, 'rb') as f:
        data = {'filename': 'notebook.ipynb'}
        files = {'file': f}
        r = requests.post(post_url, data=data, files=files)
        
    with open(requirements, 'rb') as f:
        data = {'filename': 'requirements.txt'}
        files = {'file': f}
        r = requests.post(post_url, data=data, files=files)
    
    # publish the deposition on zenodo
    pub_url = access(r_json['links']['publish'])
    import pprint
    r = requests.post(pub_url)
    pprint.pprint(r.json)