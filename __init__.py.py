"""
Human Behavior and Task Performance Modeling for AI Perception
================================================================

This package provides tools for analyzing human performance states
by integrating physiological and task performance metrics.

Modules:
    performance_states: Core state classification and analysis
    visualization: Visualization utilities
    utils: Helper functions
"""

from .performance_states import PerformanceStateAnalyzer, analyze_performance_states

__version__ = "1.0.0"
__author__ = "Your Name"
__all__ = ['PerformanceStateAnalyzer', 'analyze_performance_states']