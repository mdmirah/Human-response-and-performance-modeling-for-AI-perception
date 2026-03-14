# User Experience and Performance Modeling for AI Perception (Aerospace Implementation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![AIAA Paper](https://img.shields.io/badge/AIAA%20Scitech%202026-2734-red.svg)](https://doi.org/10.2514/6.2026-2734)

**A generic framework for analyzing user experience states by integrating physiological signals (heart rate, pupil diameter) with task performance metrics (deviation, rate of change). Initially developed for Human-AI perception and teaming for a flight task with application potential in Autonomous driving and First Person Shooter Video Games.**

I originally publishhed the theoretical framework and core methodologyin this repository in:

**Rahman, M. M., & Fala, N. (2026). "A Multimodal Framework for Assessing Fitness to Fly in Flight Training."** *AIAA Scitech 2026 Forum*, Paper AIAA 2026-2734. https://doi.org/10.2514/6.2026-2734

This paper introduces the **Fitness-to-Fly (F2Y)** score, an operational construct built on this framework that integrates a pilot's heart rate with flight performance metrics to estimate their internal state of fitness. 
**If you use this code in your research, please cite both the original paper (for the theoretical framework) and this repository (for the code implementation).**

<img width="481" height="303" alt="image" src="https://github.com/user-attachments/assets/ccef5739-2592-48c1-a064-90512f7a37ce" />

<img width="1391" height="701" alt="Screenshot 2026-03-05 104447" src="https://github.com/user-attachments/assets/5f1f51ba-5a93-455f-a3f7-2728d9c4dae3" />

<img width="600" height="437" alt="image" src="https://github.com/user-attachments/assets/41ba38e3-7651-4ade-8220-848b373966df" />

<img width="600" height="437" alt="image" src="https://github.com/user-attachments/assets/fc275ca0-33be-4772-9aaf-1b7bdaae5f94" />

<img width="705" height="838" alt="image" src="https://github.com/user-attachments/assets/66db0c26-3454-484a-be93-caf599994565" />


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
git clone https://github.com/mdmirah/Human-response-and-performance-modeling-for-AI-perception.git
cd human-response-modeling

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

### Basic Usage

```python
from human_response_modeling import PerformanceStateAnalyzer
import numpy as np

# Create analyzer instance
analyzer = PerformanceStateAnalyzer(sampling_freq=60)  # 60 Hz data

# Load your data (example with synthetic data)
time = np.arange(0, 300, 1/60)  # 5 minutes
heart_rate = 70 + 10 * np.sin(2*np.pi*time/100)
pupil = 3.5 + 0.5 * np.sin(2*np.pi*time/150)
deviation = 1000 + 500 * np.sin(2*np.pi*time/200)
rate = 10 + 5 * np.cos(2*np.pi*time/200)

# Run F2Y state analysis
results = analyzer.analyze(
    heart_rate_data=heart_rate,
    deviation_data=deviation,
    rate_of_change_data=rate,
    pupil_data=pupil,
    time_data=time,
    hr_threshold=80,  # Threshold for stress classification
    min_time_jasat=150,  # JASAT time for dynamic thresholds
    dataset_name="Sample Flight Data"
)

# Access high-quality temporal data
print(f"Analysis complete! Found {results['statistics']['transitions']['total_state_changes']} state changes.")
print(f"Time in optimal state (Relaxed Controlled): {results['statistics']['state_durations']['Relaxed Controlled']['total_duration_seconds']:.2f}s")

# Extract transition matrix for AI training
transition_matrix = results['statistics']['transitions']['transition_counts']

# Extract episode characteristics for each state
for state in results['statistics']['state_durations']:
    stats = results['statistics']['state_durations'][state]
    print(f"{state}: {stats['num_episodes']} episodes, "
          f"avg duration: {stats['avg_duration_seconds']:.2f}s, "
          f"total: {stats['total_duration_seconds']:.2f}s")
```

## 📈 Example Output and Data Structure

The framework produces rich, structured data ideal for AI training:

```python
# Sample output structure
{
    'performance_states': ['Relaxed Controlled', 'Relaxed Controlled', ...],
    'statistics': {
        'state_durations': {
            'Relaxed Controlled': {
                'num_episodes': 12,
                'avg_duration_seconds': 15.3,
                'max_duration_seconds': 45.2,
                'min_duration_seconds': 2.1,
                'total_duration_seconds': 183.6,
                'durations_seconds': [15.3, 22.1, ...]
            },
            # ... other states
        },
        'transitions': {
            'transition_counts': {
                'Relaxed Controlled → Stressed Controlled': 8,
                'Stressed Controlled → Stressed Uncontrolled': 3,
                # ... all transitions including self-transitions
            },
            'total_transitions': 1245,
            'total_state_changes': 45
        }
    }
}
```

## 🤖 Applications in Adaptive Autonomy

The validated data from this framework enables several key applications:

### 1. **Human State Prediction**
Train sequence models (LSTM, Transformers) to predict upcoming state transitions based on recent history.

### 2. **Adaptive Task Allocation**
Use real-time state classification to dynamically adjust task difficulty or automation level.

### 3. **Intervention Timing Optimization**
Identify optimal moments for AI intervention based on historical transition patterns.

### 4. **Personalized State Models**
Learn individual operator baselines and adaptation patterns for personalized assistance.

## 📚 Citation Guidelines

The Fitness-to-Fly (F2Y) framework used in this study was originally proposed by 
Rahman and Fala [1], and the implementation follows the open-source code repository 
accompanying that work [2].

[1] Rahman, M. M., and Fala, N., "A Multimodal Framework for Assessing Fitness to 
    Fly in Flight Training," AIAA Scitech 2026 Forum, AIAA Paper 2026-2734, 2026.
    doi: 10.2514/6.2026-2734

[2] Rahman, M. M., and Fala, N., "Human response and Performance Modeling for 
    AI Perception," GitHub repository, 2026. https://github.com/mdmirah/Human-response-and-performance-modeling-for-AI-perception

### BibTeX Entries

```bibtex
@inproceedings{rahman2026multimodal,
  title={A Multimodal Framework for Assessing Fitness to Fly in Flight Training},
  author={Rahman, Md Mijanur and Fala, Nicoletta},
  booktitle={AIAA Scitech 2026 Forum},
  pages={2734},
  year={2026},
  doi={10.2514/6.2026-2734}
}

@software{rahman2026f2ycode,
  author = {Rahman, Md Mijanur and Fala, Nicoletta},
  title = {Human Behavior and Task Performance Modeling for AI Perception},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/human-behavior-modeling}}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📧 Contact

For questions about the theoretical framework:
- Md Mijanur Rahman - [mzr0182@auburn.edu]

For code-related issues, please use the GitHub issue tracker.
```
t for building adaptive AI systems.
