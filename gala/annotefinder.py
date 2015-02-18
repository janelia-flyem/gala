from __future__ import absolute_import
from six.moves import zip
import numpy as np
from scipy import spatial
from matplotlib import pyplot as plt


class AnnotationFinder:
    """MPL callback to display an annotation when points are clicked.
    
    The nearest point within xtol and ytol is identified.

    Register this function like this:

    plt.scatter(xdata, ydata)
    af = AnnotationFinder(xdata, ydata, annotations)
    plt.connect('button_press_event', af)
    """

    def __init__(self, xdata, ydata, annotations, axis=None):
        self.points = np.array(zip(xdata, ydata))
        self.annotations = annotations
        self.nntree = spatial.cKDTree(self.points)
        if axis is None:
            self.axis = plt.gca()
        else:
            self.axis = axis
        self.drawn_annotations = {}
        # links to other AnnotationFinder instances
        self.links = []

    def __call__(self, event):
        if event.inaxes:
            clicked = np.array([event.xdata, event.ydata])
            if self.axis is None or self.axis == event.inaxes:
                nnidx = self.nntree.query(clicked)[1]
                x, y = self.points[nnidx]
                annotation = self.annotations[nnidx]
                self.draw_annotation(event.inaxes, x, y, annotation)
                for link in self.links:
                    link.draw_specific_annotation(annotation)

    def draw_annotation(self, axis, x, y, annotation):
        """Draw the annotation on the plot."""
        if (x, y) in self.drawn_annotations:
            markers = self.drawn_annotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            plt.axes(self.axis)
            self.axis.figure.canvas.draw()
        else:
            t = axis.text(x, y, "(%3.2f, %3.2f) - %s" % (x, y, annotation), )
            self.drawn_annotations[(x, y)] = [t]
            plt.axes(self.axis)
            self.axis.figure.canvas.draw()

    def draw_specific_annotation(self, annotation):
        to_draw = [(x, y, a)
                   for (x, y), a in zip(self.points, self.annotations)
                   if a == annotation]
        for x, y, a in to_draw:
            self.draw_annotation(self.axis, x, y, a)
