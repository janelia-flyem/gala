#!/usr/bin/env python

# Python standard library
import sys, os, argparse, cPickle

# external libraries
from numpy import unique
from scipy.ndimage.filters import median_filter

# local modules
from imio import read_image_stack, write_h5_stack, arguments as imioargs, \
    read_image_stack_single_arg
from agglo import Rag, classifier_probability, boundary_mean, random_priority, \
    approximate_boundary_mean, arguments as aggloargs
from morpho import watershed, juicy_center, arguments as morphoargs
from classify import mean_and_sem, feature_set_a, RandomForest, \
    arguments as classifyargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Segment a volume using a superpixel-to-RAG model.',
        parents=[imioargs, morphoargs, aggloargs, classifyargs]
    )
    parser.add_argument('fin', nargs='+', 
        help='The boundary probability map file(s).'
    )
    parser.add_argument('fout', 
        help='The output filename for the segmentation. Use %%str syntax.'
    )
    parser.add_argument('-P', '--show-progress', action='store_true',
        default=True, help='Show a progress bar for the agglomeration.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
        help='Print runtime information about execution.'
    )
    parser.add_argument('-o', '--objective-function', 
        metavar='FCT_NAME', default='boundary_mean',
        help='''Which merge priority function to use. Default: boundary_mean; 
        choices: boundary_mean, approximate_boundary_mean'''
    )
    args = parser.parse_args()

    if args.verbose:
        vfn = sys.stdout
    else:
        vfn = open('/dev/null', 'w')
    probs = read_image_stack(*args.fin)
    if args.invert_image:
        probs = probs.max() - probs
    if args.median_filter:
        probs = median_filter(probs, 3)
    if args.watershed is None:
        vfn.write('Computing watershed...\n')
        args.watershed = watershed(probs, show_progress=args.show_progress)
    vfn.write('Num watershed basins: %i\n'%(len(unique(args.watershed))-1))
    if args.save_watershed is not None:
        # h5py sometimes has issues overwriting files, so delete ahead of time
        if os.access(args.save_watershed, os.F_OK):
            os.remove(args.save_watershed)
        write_h5_stack(args.watershed, args.save_watershed)

    vfn.write('Computing RAG for ws and image sizes: %s, %s\n'%
        ('('+','.join(map(str,args.watershed.shape))+')',
        '('+','.join(map(str,probs.shape))+')')
    )
    if args.load_classifier is not None:
        mpf = classifier_probability(eval(args.feature_map_function), 
                                                        args.load_classifier)
    else:
        mpf = eval(args.objective_function)

    g = Rag(args.watershed, probs, show_progress=args.show_progress, 
        merge_priority_function=mpf, 
        allow_shared_boundaries=args.allow_shared_boundaries,
        lowmem=args.low_memory)

    vfn.write('RAG computed. Number of nodes: %i, Number of edges: %i\n'%
        (g.number_of_nodes(), g.number_of_edges())
    )

    if args.ladder is not None:
        if args.pre_ladder:
            vfn.write('Computing ladder agglomeration...\n')
            args.post_ladder = False
            g.agglomerate_ladder(args.ladder, args.strict_ladder)
            g.rebuild_merge_queue()
            vfn.write('Ladder done. new graph statistics: n: %i, m: %i\n'%
                (g.number_of_nodes(), g.number_of_edges())
            )
        else:
            args.post_ladder = True
    for t in args.thresholds:
        g.agglomerate(t)
        if args.ladder is not None and args.post_ladder:
            if len(args.thresholds) > 1:
                g2 = g.copy()
            else:
                g2 = g
            g2.agglomerate_ladder(args.ladder, args.strict_ladder)
        else:
            g2 = g
        try:
            write_h5_stack(g2.get_segmentation(), args.fout % t)
        except TypeError:
            write_h5_stack(g2.get_segmentation(), args.fout)
            if len(args.thresholds) > 1:
                sys.stdout.write(
                    '\nWarning: single output file but multiple thresholds '+
                    'provided. What should I do? (q: quit, first threshold '+
                    'written to file; t: give specific threshold, written '+
                    'to file; f: new filename, provide new filename for '+
                    'output.\n'
                )
                response = sys.stdin.readline()[0]
                if response == 'q':
                    break
                elif response == 't':
                    sys.stdout.write('which threshold?\n')
                    t = double(sys.stdin.readline()[:-1])
                    g.agglomerate(t)
                    if args.ladder is not None and args.post_ladder:
                        g.agglomerate_ladder(args.ladder, args.strict_ladder)
                    os.remove(args.fout)
                    write_h5_stack(g.get_segmentation(), args.fout)
                elif response == 'f':
                    args.fout = sys.stdin.readline()[:-1]
                    continue
                else:
                    sys.stdout.write('Unknown response: quitting.\n')
                    break
    g.merge_queue.finish()

