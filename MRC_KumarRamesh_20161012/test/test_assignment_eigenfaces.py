#-*- coding: utf-8 -*-
import collections
import glob
import pytest
import nbtest
import os.path
import numpy
import numpy.testing


DataForTesting = collections.namedtuple(
    'DataForTesting',
    ['training_image_filenames', 'test_image_filenames', 'train_subject_ids', 'test_subject_ids']
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

    training_image_filenames = sorted(glob.iglob(
        os.path.abspath(os.path.join(test_directory, '../data/training/*.pgm'))
    ))
    
    test_image_filenames = sorted(glob.iglob(
        os.path.abspath(os.path.join(test_directory, '../data/test/*.pgm'))
    ))

    subject_number = lambda filename: int(os.path.basename(filename)[7:9])
    train_subject_ids = map(subject_number, training_image_filenames)
    test_subject_ids = map(subject_number, test_image_filenames)

    return DataForTesting(
        training_image_filenames, test_image_filenames, train_subject_ids, test_subject_ids
    )


def test_eigenfaces(loaded_notebook, test_data):
    face_recognition = loaded_notebook.FaceRecognition()
    
    face_recognition.enroll_faces(test_data.training_image_filenames, test_data.train_subject_ids)
    recognized_ids = face_recognition.recognize_faces(test_data.test_image_filenames)

    different_results = recognized_ids - test_data.test_subject_ids
    positives = (different_results == 0).sum()
    
    assert positives >= 41
