# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 2021

@author: Gerd Duscher
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys

sys.path.append("../../sidpy/")
from sidpy.io import interface_utils


class TestInterface(unittest.TestCase):

    def test_open_file_dialog(self):
        file_widget = interface_utils.open_file_dialog()
        self.assertTrue(file_widget.file_name == '')
