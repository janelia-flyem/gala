#!/usr/bin/env python

import os, sys, argparse
import h5py
from numpy import array

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('HDF5 cat options')
arggroup.add_argument('-g', '--group', metavar='GROUP',
    help='Preview only path given by GROUP'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Preview the contents of an HDF5 file',
        parents=[arguments]
    )
    parser.add_argument('fin', help='The input HDF5 file.')

    args = parser.parse_args()
    f = h5py.File(args.fin)
    if args.group is not None:
        groups = [args.group]
    else:
        groups = []
        f.visit(groups.append)
    for g in groups:
        print g
        if type(f[g]) == h5py.highlevel.Dataset:
            a = array(f[g])
            print 'shape: ', a.shape
            print a
