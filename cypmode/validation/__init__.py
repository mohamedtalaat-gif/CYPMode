"""Structural validation against Boltz-2 predictions.

Deliberately no re-exports here: cypmode.validation.structures needs
Biopython (the `validate` extra); cypmode.validation.build_input and
cypmode.validation.constants don't. An eager import at package level would
force Biopython onto anything that touches this package at all, including
code that only needs the dependency-free pieces -- import from the
specific submodule instead.
"""
