import numpy as np
import itertools as it
from skimage import color, io
from matplotlib import colors as mplcolors

# Colors hand-picked from ColorBrewer
colors = ['#e34a33',  # dark orange
          '#3182bd',  # dark blue
          '#31a354',  # dark green
          '#dd1c77']  # magenta

colorconv = mplcolors.colorConverter
colors = [colorconv.to_rgb(c) for c in colors]

d = '/Users/nuneziglesiasj/projects/gala/doc/logo'

def logo_iterate(labels, image, fns=d + 'logo-%03i.png'):
    height, width = labels.shape
    background = (labels == 0)
    foreground = ~background
    counter = it.count()

    # part one: just foreground/background
    colorcombos = it.permutations(colors, 2)
    lab2 = np.zeros(labels.shape, np.uint8)
    lab2[foreground] = 1
    for cs in colorcombos:
        img = color.label2rgb(lab2, image, colors=cs)
        io.imsave(fns % next(counter), img)

    # part two: background split
    splits = np.arange(500, 1600, 100).astype(int)
    colorcombos = it.permutations(colors, 3)
    for s, cs in it.product(splits, colorcombos):
        im, lab = _split_img_horizontal(image, lab2, background, s)
        img = color.label2rgb(lab, im, colors=cs)
        io.imsave(fns % next(counter), img)

    # part three: foreground split
    colorcombos = it.permutations(colors, 3)
    for cs in colorcombos:
        img = color.label2rgb(labels, image, colors=cs)
        io.imsave(fns % next(counter), img)

    # part four: both split
    colorcombos = it.permutations(colors, 4)
    for s, cs in it.product(splits, colorcombos):
        im, lab = _split_img_horizontal(image, labels, background, s)
        img = color.label2rgb(lab, im, colors=cs)
        io.imsave(fns % next(counter), img)


def _split_img_horizontal(image, labels, background, height):
    image, labels = np.copy(image), np.copy(labels)
    lower = np.zeros_like(background)
    lower[height:, :] = True
    labels[background & lower] = labels.max() + 1
    strip = np.zeros_like(background)
    strip[height:(height+20), :] = True
    image[strip & background] = 0.16
    return image, labels
