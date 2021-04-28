import geohash2 as gh
import numpy as np


base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
          '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
          'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int)
          for x in range(0, len(base32))]
base32toBin = dict(zip(base32, binary))


# Deprecated - for compatibility purposes
class LatLonHash:

    def __init__(self, lat, lon):
        self._lat = lat
        self._lon = lon

    def to_hash(self, precision=15):
        return gh.encode(self._lat, self._lon, precision)

    def to_binary(self, precision=15):
        hashed = self.to_hash(precision)
        return np.concatenate([base32toBin[x] for x in hashed])


def geohash(lat, lon, precision=15):
    return gh.encode(lat, lon, precision)


def bin_geohash(lat, lon, precision=15):
    hashed = geohash(lat, lon, precision)
    return np.concatenate([base32toBin[x] for x in hashed])
