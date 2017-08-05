import os
import errno
import re


def sequence_from_path(path):
    m = re.search('\/sequences\/(\d\d)\/image', path)
    return int(m.group(1))


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
