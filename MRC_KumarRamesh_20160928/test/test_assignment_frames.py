#-*- coding: utf-8 -*-
import collections
import pytest
import nbtest
import os.path
import numpy
import numpy.testing


DataForTesting = collections.namedtuple(
    'TestData', ['points', 'tf_solutions', 'inverse_solutions', 'matrices',
                 'matrices_solutions', 'tf_solutions_sympy', 'inverse_solutions_sympy']
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
    test_points = numpy.load(os.path.join(test_directory, 'points_to_test.npy'))
    direct_tf_solutions = numpy.load(
        os.path.join(test_directory, 'direct_tf_solutions.npy')
    )
    inverse_tf_solutions = numpy.load(
        os.path.join(test_directory, 'inverse_tf_solutions.npy')
    )
    test_matrices = numpy.load(os.path.join(test_directory, 'matrices_to_test.npy'))
    matrices_solutions = numpy.load(
        os.path.join(test_directory, 'rotation_matrices_solutions.npy')
    )
    direct_tf_solutions_sympy = numpy.load(
        os.path.join(test_directory, 'direct_tf_solutions_sympy.npy')
    )
    inverse_tf_solutions_sympy = numpy.load(
        os.path.join(test_directory, 'inverse_tf_solutions_sympy.npy')
    )

    return DataForTesting(
        test_points, direct_tf_solutions, inverse_tf_solutions, test_matrices,
        matrices_solutions, direct_tf_solutions_sympy, inverse_tf_solutions_sympy
    )


def test_direct_transforms(loaded_notebook, test_data):
    """
    Tests that the homogeneous transform correctly describes the door's pose
    relative to the robot's base frame.

    """
    student_solution = loaded_notebook.direct_transform()

    for tp, rp in zip(test_data.points, test_data.tf_solutions):
        student_result = numpy.dot(
            student_solution, numpy.hstack((tp, 1))
        )
        # delete the extra 1 added to perform the multiplication
        student_result = numpy.delete(student_result, [3])

        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )


def test_inverse_transforms(loaded_notebook, test_data):
    """
    Tests that the homogeneous transform correctly describes the package's pose
    relative to the robot's end-effector frame.

    """
    student_solution = loaded_notebook.inverse_transform()

    for tp, rp in zip(test_data.points, test_data.inverse_solutions):
        student_result = numpy.dot(
            student_solution, numpy.hstack((tp, 1))
        )
        # delete the extra 1 added to perform the multiplication
        student_result = numpy.delete(student_result, [3])

        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )


def test_rotation_matrices(loaded_notebook, test_data):
    """
    Tests that the function 'is_rotation_matrix' returns correctly,
    whether a matrix is a rotation matrix or not.

    """
    for mat, res in zip(test_data.matrices, test_data.matrices_solutions):
        student_solution = loaded_notebook.is_rotation_matrix(mat)
        numpy.testing.assert_equal(student_solution, res)


def test_direct_transforms_sympy(loaded_notebook, test_data):
    """
    Tests that the homogeneous transform correctly describes the door's pose
    relative to the robot's base frame using SymPy.

    """
    student_solution, b_frame, d_frame = loaded_notebook.direct_transform_sympy()
    rotation = b_frame.dcm(d_frame)
    rotation = numpy.array(numpy.asarray(rotation), dtype=numpy.float64)

    for tp, rp in zip(test_data.points, test_data.tf_solutions_sympy):
        student_result = numpy.dot(rotation, tp)
        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )

    for tp, rp in zip(test_data.points, test_data.tf_solutions):
        student_result = numpy.dot(
            student_solution, numpy.hstack((tp, 1))
        )
        # delete the extra 1 added to perform the multiplication
        student_result = numpy.delete(student_result, [3])

        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )


def test_inverse_transforms_sympy(loaded_notebook, test_data):
    """
    Tests that the homogeneous transform correctly describes the package's pose
    relative to the robot's end-effector frame using SymPy.

    """
    student_solution, e_frame, p_frame = loaded_notebook.inverse_transform_sympy()
    rotation = e_frame.dcm(p_frame)
    rotation = numpy.array(numpy.asarray(rotation), dtype=numpy.float64)

    for tp, rp in zip(test_data.points, test_data.inverse_solutions_sympy):
        student_result = numpy.dot(rotation, tp)
        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )

    for tp, rp in zip(test_data.points, test_data.inverse_solutions):
        student_result = numpy.dot(
            student_solution, numpy.hstack((tp, 1))
        )
        # delete the extra 1 added to perform the multiplication
        student_result = numpy.delete(student_result, [3])

        numpy.testing.assert_array_almost_equal(
            student_result.tolist(), rp.tolist(), decimal=4
        )
