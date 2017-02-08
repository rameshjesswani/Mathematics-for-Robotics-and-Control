#-*- coding: utf-8 -*-
import collections
import pytest
import nbtest
import os.path
import numpy
import numpy.testing


DataForTesting = collections.namedtuple(
    'TestData', ['bins', 'point_cloud']
)


@pytest.fixture(scope='session')
def loaded_notebook():
    base, ext = os.path.splitext(os.path.basename(__file__))
    # remove test_ prefix
    notebook_name = base[5:]
    search_path = [os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
    loader = nbtest.NotebookLoader(path=search_path)
    notebook = loader.load_module(notebook_name)
    return notebook


@pytest.fixture(scope='session')
def test_data():
    test_directory = os.path.dirname(__file__)
    bins = numpy.load(os.path.join(test_directory, 'bins.npy'))

    point_cloud = numpy.loadtxt(
        os.path.join(test_directory, '../data/cloud.asc'), delimiter=" "
    )

    return DataForTesting(bins, point_cloud)


def test_angles_to_zaxis(loaded_notebook, test_data):
    expected_bins = test_data.bins
    bin_indices = numpy.where(expected_bins != 0xFFFFFFFF)
    student_angles = loaded_notebook.rate_placements(test_data.point_cloud)
    student_bins = numpy.digitize(student_angles, numpy.linspace(0, 2*numpy.pi, 250))
    D = numpy.abs(student_bins[bin_indices] - expected_bins[bin_indices])
    assert numpy.all(D <= 1)

