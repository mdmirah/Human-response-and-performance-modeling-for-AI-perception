{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Human Behavior and Task Performance Modeling\n",
    "## Example Usage Notebook"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from human_behavior_modeling import PerformanceStateAnalyzer\n",
    "\n",
    "# Generate sample data\n",
    "np.random.seed(42)\n",
    "time = np.arange(0, 300, 1/60)  # 5 minutes at 60 Hz\n",
    "\n",
    "# Simulate pilot data\n",
    "heart_rate = 70 + 10 * np.sin(2*np.pi*time/100) + np.random.normal(0, 2, len(time))\n",
    "pupil = 3.5 + 0.5 * np.sin(2*np.pi*time/150) + np.random.normal(0, 0.1, len(time))\n",
    "deviation = 1000 + 500 * np.sin(2*np.pi*time/200) + np.random.normal(0, 50, len(time))\n",
    "rate = 10 + 5 * np.cos(2*np.pi*time/200) + np.random.normal(0, 1, len(time))\n",
    "\n",
    "# Create analyzer\n",
    "analyzer = PerformanceStateAnalyzer(sampling_freq=60)\n",
    "\n",
    "# Run analysis\n",
    "results = analyzer.analyze(\n",
    "    heart_rate_data=heart_rate,\n",
    "    deviation_data=deviation,\n",
    "    rate_of_change_data=rate,\n",
    "    pupil_data=pupil,\n",
    "    time_data=time,\n",
    "    hr_threshold=80,\n",
    "    min_time_jasat=150,\n",
    "    dataset_name=\"Sample Flight Data\"\n",
    ")\n",
    "\n",
    "# Access results\n",
    "print(f\"\\nAnalysis complete! Found {results['statistics']['transitions']['total_state_changes']} state changes.\")"
   ]
  }
 ]
}