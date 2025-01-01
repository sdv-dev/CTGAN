#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration tests for ctgan.

These tests only ensure that the software does not crash and that
the API works as expected in terms of input and output data formats,
but correctness of the data values and the internal behavior of the
model are not checked.
"""

import tempfile as tf

import numpy as np
import pandas as pd
import pytest
import torch

from ctgan.synthesizers.ctgan import CTGAN

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_ctgan_fit_sample_apple_mps_hardware(tmpdir, train_data, random_state):
    """Test the CTGAN can fit and sample."""
    ctgan = CTGAN(cuda=False, epochs=1)
    ctgan.set_random_state(random_state)
    ctgan.fit(train_data)
    sampled = ctgan.sample(1000)
    assert sampled.shape == (1000, train_data.shape[1])

    # Save and load
    path = os.path.join(tmpdir, 'test_ctgan.pkl')
    ctgan.save(path)
    ctgan = CTGAN.load(path)

    sampled = ctgan.sample(1000)
    assert sampled.shape == (1000, train_data.shape[1])



@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_training_apple_mps_hardware(tmpdir, train_data, random_state):
    """Test CTGAN training on MPS device."""
    ctgan = CTGAN(cuda=False, mps=True, epochs=1)
    ctgan.set_random_state(random_state)

    # Check device of model components before training
    assert ctgan._device.type == 'mps'
    assert next(ctgan._generator.parameters()).device.type == 'mps'

    ctgan.fit(train_data)

    # Check device of model components after training
    assert next(ctgan._generator.parameters()).device.type == 'mps'

    sampled = ctgan.sample(100)
    assert sampled.shape == (100, train_data.shape[1])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_save_load_apple_mps_hardware(tmpdir, train_data, random_state):
    """Test the CTGAN saves and loads correctly."""
    ctgan = CTGAN(cuda=False, epochs=1)
    ctgan.set_random_state(random_state)
    ctgan.fit(train_data)

    # Save and load
    path = os.path.join(tmpdir, 'test_ctgan.pkl')
    ctgan.save(path)
    ctgan = CTGAN.load(path)

    # Check device type after loading
    if torch.backends.mps.is_available():
        assert ctgan._device.type == 'mps'
        assert next(ctgan._generator.parameters()).device.type == 'mps'
    elif torch.cuda.is_available():
        assert ctgan._device.type == 'cuda'
        assert next(ctgan._generator.parameters()).device.type == 'cuda'
    else:
        assert ctgan._device.type == 'cpu'
        assert next(ctgan._generator.parameters()).device.type == 'cpu'

    sampled = ctgan.sample(1000)
    assert sampled.shape == (1000, train_data.shape[1])