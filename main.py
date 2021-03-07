import numpy as np
import argparse
import matplotlib.pyplot as plt
import geojson

from data_base import DataBase

r_e = 6371 # km


def distance2phi(d):
    """ Convert distance to phi angle """
    return d / r_e


def height2radius(h):
    """ Convert height to radius from center of Earth """
    return r_e + h


def get_beta(phi, h, h_0=0.566):
    """ Returns Beta angle as defined to the local tangential vector beta = (90° - zenith) """
    r = height2radius(h)
    r_0 = height2radius(h_0)
    beta = np.arccos( np.sin(phi) / np.sqrt(np.sin(phi)**2 + (np.cos(phi) - r_0 / r)**2) )
    return np.sign(r * np.cos(phi) - r_0) * beta


def get_alpha(ra, dec, ra_0, dec_0):
    """ returns azimuth angle in horizontal coordinates as seen from the observer """
    a = np.abs(ra - ra_0) * np.cos(np.minimum(dec, dec_0))  # angular distance alonge equatorial right ascension
    b = np.abs(dec - dec_0)
    cos_c = np.cos(a) * np.cos(b)
    return np.sign(ra - ra_0) * np.arccos(np.cos(a) * np.sin(b) / np.sqrt(1 - cos_c**2))


def angle(v1, v2):
    """
    Angular distance in radians for each pair from two (lists of) vectors.
    Use each2each=True to calculate every combination.

    :param v1: vector(s) of shape (3, n)
    :param v2: vector(s) of shape (3, n)
    :return: angular distance in radians
    """
    if len(v1.shape) == 1:
        v1 = v1.reshape(3, 1)
    if len(v2.shape) == 1:
        v2 = v2.reshape(3, 1)
    d = np.sum(v1 * v2, axis=0)
    return np.arccos(np.clip(d, -1., 1.))


def ang2vec(phi, theta):
    """
    Get vector from spherical angles (phi, theta)

    :param phi: range (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    :param theta: range (pi/2, -pi/2), pi/2 points in z-direction
    :return: vector of shape (3, n)
    """
    assert np.ndim(phi) == np.ndim(theta), "Inputs phi and theta in 'coord.ang2vec()' must have same shape!"
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ra_0', '--ra_0', default=11.521400, type=float, help='Right ascension of observer.')
    parser.add_argument('-dec_0', '--dec_0', default=48.110518, type=float, help='Declination of observer.')
    parser.add_argument('-h_0', '--h_0', default=0.566, type=float, help='Height of observer.')
    parser.add_argument('-osm', '--osm', default='data/gap_150.geojson', type=str, help='Library from OSM.')
    parser.add_argument('-map', '--map', default=False, action='store_true', help='Plot a map.')
    kw = parser.parse_args()

    # v refers to a unit vector pointing from the center of Earth to the geographic coordinate
    v_ref = ang2vec(np.deg2rad(kw.ra_0), np.deg2rad(kw.dec_0))

    db = DataBase()
    db.update_database()
    n = db.get('n')

    coords, ele, name = db.get('coordinates'), db.get('ele'), db.get('name')
    if kw.map:
        plt.scatter(coords[:, 0], coords[:, 1], marker='^', c='k', s=ele/500.)
        plt.scatter(kw.ra_0, kw.dec_0, color='red', marker='o', s=30)
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.show()

    v = ang2vec(*np.deg2rad(coords.T))
    phi = angle(v, v_ref)

    beta = np.rad2deg(get_beta(phi, ele / 1000., h_0=kw.h_0))
    print('elevation (min, median, max): ', np.min(ele), np.median(ele), np.max(ele))
    print('\nHighest peaks in data base: ')
    for i, idx in enumerate(np.argsort(ele)[::-1][:9]):
        print('\t%i) %s (%s m)' % (i+1, name[idx], ele[idx]))

    print('\nMost visible peaks from observer: ')
    print(beta.shape)
    for i, idx in enumerate(np.argsort(beta)[::-1][:9]):
        print('\t%i) %s (%s m, beta=%.2f deg) @ (%s, %s)' % (i+1, np.array(name)[idx], ele[idx], beta[idx], coords[idx][1], coords[idx][0]))
    idx_most_north = np.argmax(coords[:, 1])

    alpha = np.rad2deg(get_alpha(np.deg2rad(coords[:, 0]), np.deg2rad(coords[:, 1]), np.deg2rad(kw.ra_0), np.deg2rad(kw.dec_0)))

    plot_mask = np.ones(n).astype(bool)
    plot_mask[beta <= 0] = False
    for idx in np.argsort(phi):
        if plot_mask[idx]:
            plt.scatter(-alpha[idx], beta[idx], marker="^", color='k', s=20)
        plot_mask[beta < beta[idx] - 0.5 * np.abs(alpha - alpha[idx])] = False
    ra_church, dec_church, h_church = np.deg2rad(11.517825966776131), np.deg2rad(48.077290623602025), 0.637
    alpha_church = get_alpha(ra_church, dec_church, np.deg2rad(kw.ra_0), np.deg2rad(kw.dec_0))
    beta_church = get_beta(angle(ang2vec(ra_church, dec_church), v_ref), h_church, h_0=kw.h_0)
    plt.scatter(-np.rad2deg(alpha_church), np.rad2deg(beta_church), marker='+', color='red', s=30)
    print('\nNumber of plotted peaks: ', np.sum(plot_mask))
    plt.xlabel('azimuth / deg')
    plt.ylabel('altitude / deg')
    plt.show()
