import os
import time
from urllib.parse import urlparse

import fire
import requests


def parse_url(url):
    result = urlparse(url)
    if result.scheme == 'https':
        assert result.netloc == 'openaipublic.blob.core.windows.net'
        return result.path.lstrip('/')
    else:
        raise Exception(f'Could not parse {url} as an Azure url')


def download_file_cached(url, cache_dir='/tmp/azure-cache'):
    """ Given an Azure path url, caches the contents locally.
        WARNING: only use this function if contents under the path won't change!
        """
    path = parse_url(url)
    filename = '_'.join(path.rsplit('/')[-2:])  # path is like 'lm-human-preferences/labels/tldr/online_45k.json'
    local_path = os.path.join(cache_dir, filename)
    sentinel = local_path + '.SYNCED'
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print('Downloading training labels for reward model')
        r = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192 * 8):
                f.write(chunk)

        open(sentinel, 'a').close()
    return local_path


if __name__ == '__main__':
  fire.Fire(download_file_cached)
