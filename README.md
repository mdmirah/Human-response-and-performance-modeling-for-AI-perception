# Human Behavior and Task Performance Modeling for AI Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for analyzing human performance states by integrating physiological signals (heart rate, pupil diameter) with task performance metrics (deviation, rate of change). Designed for aviation psychology research and AI-based operator state monitoring.

## 🎯 Key Features

- **Multi-dimensional State Classification**: Combines physiological and performance metrics for holistic operator state assessment
- **Dynamic Thresholding**: Adapts performance thresholds based on JASAT (Just Another Source of Action Time)
- **Comprehensive Statistics**: Calculates state durations, transitions, and episode analysis
- **Rich Visualizations**: Generates publication-quality plots with state-based coloring
- **Self-Transition Tracking**: Includes same-state persistence in transition analysis

## 📊 State Classification Logic

The system classifies operator states along two dimensions:

### Physiological State (HR/PD)
- **Stressed**: High Heart Rate AND High Pupil Diameter
- **Relaxed**: Any other combination

### Performance State
- **Controlled**: Low Deviation AND Low Rate of Change (Smooth)
- **Uncontrolled**: Any other combination

### Combined States
1. **Stressed Uncontrolled** 🔴
2. **Relaxed Uncontrolled** 🟠
3. **Stressed Controlled** 🔵
4. **Relaxed Controlled** 🟢

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/human-behavior-modeling.git
cd human-behavior-modeling

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
