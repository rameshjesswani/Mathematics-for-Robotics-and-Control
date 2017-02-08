#!/usr/bin/env python
#-*- coding: utf-8 -*-

import collections
import control
import functools
import matplotlib.pyplot as plt
import numpy
import numpy.linalg
#import scipy.spatial
import sys

RootLocusResult = collections.namedtuple("RootLocusResult", [
    "pole", "gain", "damping_coefficient", "overshoot", "frequency"
])

def angle_between_points_on_complex_plane(p1, p2):
    origin_x = p2.real
    origin_y = p2.imag
    point_x = p1.real
    point_y = p1.imag
    return numpy.degrees(numpy.arctan2(point_y - origin_y, point_x - origin_x))


def distance_between_points_on_complex_plane(p1, p2):
    origin_x = p2.real
    origin_y = p2.imag
    point_x = p1.real
    point_y = p1.imag
    return numpy.linalg.norm([point_x - origin_x, point_y - origin_y], 2)


def point_is_on_root_locus(tf, point):
    poles = tf.pole()
    zeros = tf.zero()
    angle_to_point = functools.partial(
        angle_between_points_on_complex_plane, point
    )
    pole_angles = map(angle_to_point, poles)
    zero_angles = map(angle_to_point, zeros)
    angle_diff = sum(zero_angles) - sum(pole_angles)
    angle_diff_is_multiple_of_180 = (angle_diff % 180) == 0
    if not angle_diff_is_multiple_of_180:
        return False
    angle_diff = int(angle_diff)
    angle_diff_is_uneven_multiple_of_180 = bool((angle_diff // 180) & 1)
    return angle_diff_is_uneven_multiple_of_180


def gain_at(tf, point):
    poles = tf.pole()
    zeros = tf.zero()
    length_to_point = functools.partial(
        distance_between_points_on_complex_plane, point
    )
    pole_lengths = map(length_to_point, poles)
    zero_lengths = map(length_to_point, zeros)
    return numpy.product(pole_lengths) / numpy.product(zero_lengths)


def damping_coefficient_at(point):
    return -numpy.cos(numpy.arctan2(point.imag, point.real))


def undamped_natural_frequency_at(point):
    return numpy.linalg.norm([point.real, point.imag], 2)


def overshoot_from_damping_coefficient(damping_coefficient):
    return 100. * numpy.exp(
        (-damping_coefficient*numpy.pi) / numpy.sqrt(1. - damping_coefficient**2)
    )


def rlocusfind(system_or_tf, *args, **kwargs):
    kwargs["Plot"] = False
    initial_pick = None
    if "initial_pick" in kwargs:
        initial_pick = kwargs["initial_pick"]
        del kwargs["initial_pick"]
    
    plot_str = kwargs["plotstr"] if "plotstr" in kwargs else "-"
    rl, gains = control.rlocus(system_or_tf, *args, **kwargs)
    
    x = numpy.concatenate(numpy.real(rl.T))
    y = numpy.concatenate(numpy.imag(rl.T))
    #loci_points = numpy.squeeze(numpy.dstack( (x, y) ))
    #lu_tree = scipy.spatial.KDTree(loci_points, 5)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def rlocus_result_at(pole_x, pole_y):
        pole = complex(pole_x, pole_y)
        
        gain = gain_at(system_or_tf, pole)
        damping_coefficient = damping_coefficient_at(pole)
        overshoot = overshoot_from_damping_coefficient(damping_coefficient)
        natural_frequency = undamped_natural_frequency_at(pole)

        return RootLocusResult(
            pole, gain, damping_coefficient, overshoot, natural_frequency
        )

        
    def pick_handler(event):
        try:
            mouse_event = event.mouseevent
            locus_line = event.artist
            #dist, point_index = lu_tree.query(
            #    (mouse_event.xdata, mouse_event.ydata)
            #)
            
            pole_x = mouse_event.xdata #x[point_index]
            pole_y = mouse_event.ydata #y[point_index]
            marker.set_xdata([pole_x])
            marker.set_ydata([pole_y])
            fig.canvas.draw()

            res = rlocus_result_at(pole_x, pole_y)
            
            print "-"*25
            print "System:\n{0!s}".format(system_or_tf)
            print "Gain:", res.gain
            print "Pole: {0!s}".format(res.pole)
            print "Damping:", res.damping_coefficient 
            print "Overshoot (%):", res.overshoot
            print "Frequency (rad/s):", res.frequency
            print "-"*25
        except Exception as e:
            print "Error: {0!s}".format(e)
        sys.stdout.flush()

    if not initial_pick:
        rlocus_result = None
        marker, = ax.plot([x[0]], [y[0]], "ko")
    else:
        initial_x, initial_y = initial_pick
        rlocus_result = rlocus_result_at(initial_x, initial_y)
        marker, = ax.plot([initial_x], [initial_y], "ko")
        SynthesizedEvent = collections.namedtuple(
            "SynthesizedEvent", ["mouseevent", "artist"]
        )
        artificial_pick_event = SynthesizedEvent( initial_pick, None )
        
    for col in rl.T:
        ax.plot(numpy.real(col), numpy.imag(col), plot_str, picker=3)

    ax.grid(True)    
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    
    fig.canvas.mpl_connect("pick_event", pick_handler)
    return (fig, rlocus_result)

