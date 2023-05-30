"""Gradients allow you to experiment with CFG and strength over several pics."""
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda" # TODO: Add float16 everywhere when CUDA


class Gradient(Enum):
    NoGradient = 0
    CFGGradient = 1
    StepsGradient = 2
    StrengthGradient = 3


RANGES = {
    Gradient.NoGradient: 100,  # No gradient, so 100 pics generated until stopped
    Gradient.CFGGradient: 20,  # CFG to range from 0 to 19
    Gradient.StepsGradient: 101,
    Gradient.StrengthGradient: 10,  # strength to range from 1 to 10
}
