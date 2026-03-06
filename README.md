# Human Response and Performance Modeling for AI Perception

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![AIAA Paper](https://img.shields.io/badge/AIAA%20Scitech%202026-2734-red.svg)](https://doi.org/10.2514/6.2026-2734)

**A framework for analyzing human performance states by integrating physiological signals (heart rate, pupil diameter) with task performance metrics (deviation, rate of change). Initially developed for Human-AI perception and teaming for a flight task with application potential in Autonomous driving and First Person Shooter Video Games.**

I originally publishhed the theoretical framework and core methodologyin this repository in:

**Rahman, M. M., & Fala, N. (2026). "A Multimodal Framework for Assessing Fitness to Fly in Flight Training."** *AIAA Scitech 2026 Forum*, Paper AIAA 2026-2734. https://doi.org/10.2514/6.2026-2734

This paper introduces the **Fitness-to-Fly (F2Y)** score, an operational construct built on this framework that integrates a pilot's heart rate with flight performance metrics to estimate their internal state of fitness. 
**If you use this code in your research, please cite both the original paper (for the theoretical framework) and this repository (for the code implementation).**

<img width="1391" height="701" alt="Screenshot 2026-03-05 104447" src="https://github.com/user-attachments/assets/5f1f51ba-5a93-455f-a3f7-2728d9c4dae3" />

## 🎯 Key Features

- **Multi-dimensional State Classification**: Combines physiological and performance metrics for holistic operator state assessment based on the F2Y framework
- **Dynamic Thresholding**: Adapts performance thresholds based on critical points (The JASAT waypoint in this implementation) 
- **Comprehensive Statistics**: Calculates state durations, transitions, and episode analysis
- **Rich Visualizations**: Generates publication-quality plots with state-based coloring
- **State-Transition Tracking**: Includes state transition and persistence in transition analysis
- **Statistical Validation**: Produces high-quality, statistically significant data with inherent structure

## 📊 Human State Classification Logic

The system classifies operator states along two dimensions as defined in Rahman & Fala (2026):

### Physiological State (HR/PD)
- **Stressed**: High Heart Rate AND High Pupil Diameter (Can be retrofitted with any required human response feature and directional relationship)
- **Relaxed**: Any other combination

### Performance State
- **Controlled**: Low Deviation AND Low Rate of Change (Smooth) (Can be retrofitted with any required task performance feature and directional relationship)
- **Uncontrolled**: Any other combination

### Combined F2Y States
1. **Stressed Uncontrolled** 🔴 - High physiological arousal with poor performance
2. **Relaxed Uncontrolled** 🟠 - Low physiological arousal with poor performance  
3. **Stressed Controlled** 🔵 - High physiological arousal with good performance
4. **Relaxed Controlled** 🟢 - Low physiological arousal with good performance (Ideal state)

## 📊 Data Quality and Validation

This framework generates **high-quality behavioral data** with demonstrated statistical significance:

- **Temporal Precision**: Captures exact time spent in each of the four F2Y states at 60 Hz sampling rate
- **Transition Dynamics**: Quantifies transition counts and patterns between states, revealing operator state evolution
- **Episode Analysis**: Provides detailed statistics on state durations, including average, minimum, and maximum episode lengths
- **Statistical Structure**: The extracted features (state durations, transition matrices, episode characteristics) exhibit inherent statistical structure that has been validated through rigorous analysis

The framework's ability to produce statistically significant data makes it ideal for:
- Training machine learning models for state prediction
- Developing adaptive autonomy systems
- Establishing baselines for operator performance
- Identifying critical transitions in human-AI teaming scenarios including Aviation, Driving, and FPS gaming.

## 🧠 Foundation for Human-AI Perception

This framework and the high-quality data it produces lay the essential groundwork for **Human-AI perception systems** that can:

1. **Real-time State Awareness**: Enable AI systems to perceive and track human operator states continuously
2. **Adaptive Autonomy**: Develop systems that adjust their level of automation based on detected operator state
3. **Predictive Intervention**: Anticipate performance degradation before it occurs
4. **Personalized Assistance**: Tailor support based on individual operator state patterns

The statistically validated data structure provides the foundation for training robust AI models that can:
- Recognize operator state from physiological and performance metrics
- Predict state transitions before they occur
- Optimize timing for automation mode switches
- Learn individual operator patterns for personalized assistance

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
