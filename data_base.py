import os
import numpy as np
import pickle
import glob
import hashlib
import geojson


def clean_name(name, key='ele'):
    """ Cleans typical file name mismatches from OSM database """

    if key == 'ele':
        if name.endswith(' m'):
            name = name.replace(' m', '')   # cut meter unit at end
        if name.startswith('~'):
            name = name.replace('~', '')    # approximate in the beginning
        if (',' in name):
            name = name.replace(',', '.')   # german decimal
        if (name == ''):
            name = '0'

    return name


def dec2hms(val):
    h = int(val)
    m = int(60 * (val - h))
    s = round(3600 * (val - h - m / 60.), 1)
    return "{}°{}'{}\"".format(h, m, s)


def get_osm_key(data, key):
    """ Find correct class of key in OSM file """
    return data['geometry'][key] if key == 'coordinates' else data['properties'][key]


def get_hash(fname, BLOCK_SIZE=65536):

    file_hash = hashlib.sha256()
    with open(fname, 'rb') as f:
        fb = f.read(BLOCK_SIZE)
        while len(fb) > 0:
            file_hash.update(fb)
            fb = f.read(BLOCK_SIZE)

    return file_hash.hexdigest()


class DataBase(dict):

    def __init__(self, fname=None):

        self.__file__ = 'data/database.pkl' if fname is None else fname
        self.must_keys = ['osm_id', 'coordinates', 'ele']
        self.all_keys = ['osm_id', 'name', 'name:de', 'coordinates', 'ele', 'prominence']

        self.update({'n': 0})
        self.update({'hashes': []})
        if os.path.exists(self.__file__):
            self.load()

    def update_database(self, force=False):

        _updated = False
        geojson_files = glob.glob('data/*.geojson')
        for fname in geojson_files:
            _hash = get_hash(fname)
            if (_hash in self.get('hashes')) and (force is False):
                print('Skip file %s (%s): already in database' % (fname, _hash))
                continue
            elif force is False:
                self.update({'hashes': self.get('hashes') + [_hash]})
            data = geojson.load(open(fname))['features']
            _n = len(data)

            pick = np.zeros(_n).astype(bool)
            osm_id = np.zeros(_n).astype(int)
            coordinates, ele, prominence = np.zeros((_n, 2)), np.zeros(_n), np.zeros(_n)
            name, name_de = np.array(_n * [''], dtype='<U32'), np.array(_n * [''], dtype='<U32')
            for i in range(_n):
                if not self.check_must_keys(data[i]):
                    continue
                osm_id[i] = get_osm_key(data[i], 'osm_id')
                coordinates[i] = get_osm_key(data[i], 'coordinates')
                ele[i] = float(clean_name(get_osm_key(data[i], 'ele')))
                prominence[i] = float(clean_name(get_osm_key(data[i], 'prominence')))
                name[i] = get_osm_key(data[i], 'name')
                name_de[i] = get_osm_key(data[i], 'name:de')
                pick[i] = True

            in_database = np.in1d(osm_id, self.get('osm_id'))
            pick = pick & ~in_database
            n_pick = np.sum(pick)
            print('Take %i / %i items' % (n_pick, _n))

            self.update({'osm_id': np.concatenate((self.get('osm_id', []), osm_id[pick])),
                         'coordinates': np.concatenate((self.get('coordinates'), coordinates[pick])) if 'coordinates' in self.keys() else coordinates[pick],
                         'ele': np.concatenate((self.get('ele', []), ele[pick])),
                         'prominence': np.concatenate((self.get('prominence', []), prominence[pick])),
                         'name': np.concatenate((self.get('name', []), name[pick])),
                         'name_de': np.concatenate((self.get('name_de', []), name_de[pick]))})
            self.update({'n': self.get('n') + n_pick})
            assert len(self.get('osm_id')) == self.get('n'), "Error for length of OSM file"
            _updated = True

        if _updated:
            self.save()

    def save(self):
        """ Save database """
        if os.path.exists(self.__file__):
            os.rename(self.__file__, self.__file__.replace('.pkl', '_bak.pkl'))
        with open(self.__file__, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.__file__, 'rb') as f:
            self.update(pickle.load(f))

    def print_osm_item(self, data, i):

        print('\ngeometry')
        geom = data[i]['geometry']
        for key in geom.keys():
            if (geom[key] != '') and (len(geom[key]) > 0):
                print('\t%s: ' % key, data[i]['geometry'][key])
        print('\nproperties')
        prop = data[i]['properties']
        for key in prop.keys():
            if (prop[key] != '') and (len(prop[key]) > 0):
                print('\t%s: ' % key, prop[key])

    def check_must_keys(self, data):

        for key in self.must_keys:
            if get_osm_key(data, key) == '':
                return False

        return True

    def label_by(self, key='ele'):
        import webbrowser
        _name, _coords, key_val = self.get('name'), self.get('coordinates'), self.get(key)
        _url = 'https://www.google.com/maps/place/'
        for i, idx in enumerate(np.argsort(key_val)[::-1]):
            if len(_name[idx]) == 0:
                lon, lat = _coords[idx]
                url = _url + "{}N+{}E".format(dec2hms(lat), dec2hms(lon))
                webbrowser.open(url, new=2)
                inp = input("Enter name (%s=%s): " % (key, key_val[idx]))
                if inp in ['break', 'stop', 'exit']:
                    break
                _name[idx] = inp
        self.update({'name': _name})
        self.save()


if __name__ == '__main__':

    db = DataBase()
    db.update_database()
    db.label_by()
    # db.save()
