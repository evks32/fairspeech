import json
import urllib.request
from pathlib import Path
import win32cred

cred = win32cred.CredRead('GitHub - https://api.github.com/evks32', win32cred.CRED_TYPE_GENERIC)
token = cred['CredentialBlob'].decode('utf-8')

# 1. Get release info
headers = {'Authorization': f'token {token}', 'User-Agent': 'FairSpeech-App'}
req = urllib.request.Request(
    'https://api.github.com/repos/evks32/fairspeech/releases/tags/v0.1-asr-outputs',
    headers=headers
)
with urllib.request.urlopen(req) as resp:
    rel = json.loads(resp.read().decode())

rel_id = rel['id']
print(f'Release ID: {rel_id}')

# 2. Delete existing assets if any
for a in rel.get('assets', []):
    asset_id = a['id']
    asset_name = a['name']
    print(f'Deleting old asset {asset_name} (ID: {asset_id})...')
    del_req = urllib.request.Request(
        f'https://api.github.com/repos/evks32/fairspeech/releases/assets/{asset_id}',
        headers=headers,
        method='DELETE'
    )
    with urllib.request.urlopen(del_req) as resp:
        print(f'Deleted {asset_name}: {resp.status}')

# 3. Upload new assets
upload_url_base = f'https://uploads.github.com/repos/evks32/fairspeech/releases/{rel_id}/assets'

files_to_upload = [
    Path('data/processed/whisper_outputs.zip'),
    Path('data/processed/asr_and_human_outputs.zip'),
]

for p in files_to_upload:
    size = p.stat().st_size
    print(f'Uploading {p.name} ({round(size / 1024 / 1024, 2)} MB)...')
    with open(p, 'rb') as f:
        data = f.read()

    upload_url = f'{upload_url_base}?name={p.name}'
    up_req = urllib.request.Request(
        upload_url,
        data=data,
        headers={
            'Authorization': f'token {token}',
            'Content-Type': 'application/zip',
            'User-Agent': 'FairSpeech-App'
        }
    )
    with urllib.request.urlopen(up_req) as resp:
        print(f'Uploaded {p.name} successfully: {resp.status}')

print('All assets uploaded successfully!')
