import os
import urllib
from tqdm import tqdm

PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.realpath(os.path.join(PROJECT_PATH, "data"))
RESULTS_PATH = os.path.realpath(os.path.join(PROJECT_PATH, "results"))
CONFIG_PATH = os.path.realpath(os.path.join(PROJECT_PATH, "config"))


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
