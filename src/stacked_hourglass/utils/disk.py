
from diskcache import FanoutCache


def getCache(scope_str):
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       shards=64,
                       timeout=1,
                       )