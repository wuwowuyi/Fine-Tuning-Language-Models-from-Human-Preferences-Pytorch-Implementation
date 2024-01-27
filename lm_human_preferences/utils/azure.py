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


def download_file_cached(url, comm=None):
    """ Given an Azure path url, caches the contents locally.
        WARNING: only use this function if contents under the path won't change!
        """
    cache_dir = '/tmp/azure-cache'
    path = parse_url(url)
    is_master = not comm or comm.Get_rank() == 0
    local_path = os.path.join(cache_dir, path)

    sentinel = local_path + '.SYNCED'
    if is_master:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            r = requests.get(url, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 8):
                    f.write(chunk)

            open(sentinel, 'a').close()
    else:
        while not os.path.exists(sentinel):
            time.sleep(1)
    return local_path


if __name__ == '__main__':
  fire.Fire(download_file_cached)
