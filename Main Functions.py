"""
Human Behavior and Task Performance Modeling for AI Perception
================================================================
This module analyzes pilot performance states by integrating physiological 
(heart rate, pupil diameter) and task performance (deviation, rate of change) 
metrics to classify operator states.

Author: Md Mijanur rahman (Mijan Rahman)
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


class PerformanceStateAnalyzer:
    """
    Analyzes pilot performance states based on physiological and task metrics.
    
    This class implements a multi-dimensional state classification system that
    combines:
    - Physiological state: Stressed/Relaxed based on HR and pupil diameter
    - Performance state: Controlled/Uncontrolled based on deviation and rate of change
    
    The analyzer supports dynamic thresholds based on JASAT (Just Another Source 
    of Action Time) for adaptive performance evaluation.
    """
    
    # Default thresholds (can be overridden)
    DEFAULT_DEV_THRESHOLD = 1192.58  # ft
    DEFAULT_RATE_THRESHOLD = 14.08   # ft/s
    
    # Dynamic thresholds based on empirical data
    DEV_THRESHOLD_BEFORE_JASAT = 1921.54  # ft
    DEV_THRESHOLD_AFTER_JASAT = 602.09    # ft
    RATE_THRESHOLD_BEFORE_JASAT = 18.67   # ft/s
    RATE_THRESHOLD_AFTER_JASAT = 13.90    # ft/s
    
    # State color mapping for consistent visualization
    STATE_COLORS = {
        "Stressed Uncontrolled": "#FF0000",  # Red
        "Relaxed Uncontrolled": "#FFA500",   # Orange
        "Stressed Controlled": "#0000FF",     # Blue
        "Relaxed Controlled": "#008000"       # Green
    }
    
    def __init__(self, sampling_freq: float = 60.0):
        """
        Initialize the performance state analyzer.
        
        Args:
            sampling_freq: Sampling frequency in Hz (default: 60 Hz)
        """
        self.sampling_freq = sampling_freq
        self._validate_sampling_freq()
        
    def _validate_sampling_freq(self):
        """Validate sampling frequency is positive."""
        if self.sampling_freq <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_freq}")
    
    def analyze(self, 
                heart_rate_data: Union[List, np.ndarray],
                deviation_data: Union[List, np.ndarray],
                rate_of_change_data: Union[List, np.ndarray],
                pupil_data: Union[List, np.ndarray],
                time_data: Union[List, np.ndarray],
                hr_threshold: float,
                min_time_jasat: Optional[float] = None,
                dataset_name: str = "",
                use_dynamic_thresholds: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive performance state analysis.
        
        Args:
            heart_rate_data: Heart rate measurements (bpm)
            deviation_data: Flight path deviation measurements (ft)
            rate_of_change_data: Rate of change of deviation (ft/s)
            pupil_data: Pupil diameter measurements (mm)
            time_data: Time stamps (seconds)
            hr_threshold: Heart rate threshold for stress classification (bpm)
            min_time_jasat: JASAT time for dynamic thresholds (seconds)
            dataset_name: Name of the dataset for reporting
            use_dynamic_thresholds: Whether to use dynamic thresholds based on JASAT
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        # Align and validate input data
        aligned_data = self._align_and_validate_data(
            heart_rate_data, deviation_data, rate_of_change_data, 
            pupil_data, time_data
        )
        
        # Unpack aligned data
        heart_rate = aligned_data['heart_rate']
        deviation = aligned_data['deviation']
        rate_of_change = aligned_data['rate_of_change']
        pupil_diameter = aligned_data['pupil_diameter']
        time = aligned_data['time']
        n_samples = aligned_data['n_samples']
        
        # Print analysis header
        self._print_analysis_header(n_samples, dataset_name)
        
        # Calculate thresholds
        thresholds = self._calculate_thresholds(
            pupil_diameter, hr_threshold, min_time_jasat, use_dynamic_thresholds
        )
        
        # Print threshold summary
        self._print_threshold_summary(thresholds, hr_threshold)
        
        # Classify states
        performance_states, hr_pd_states = self._classify_states(
            heart_rate, deviation, rate_of_change, pupil_diameter, 
            time, thresholds
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(
            performance_states, hr_pd_states, time, thresholds
        )
        
        # Generate visualizations
        self._plot_results(
            time, heart_rate, deviation, rate_of_change, pupil_diameter,
            performance_states, thresholds
        )
        
        # Compile results
        results = self._compile_results(
            performance_states, hr_pd_states, statistics, thresholds,
            heart_rate, deviation, rate_of_change, pupil_diameter, time
        )
        
        return results
    
    def _align_and_validate_data(self, *arrays) -> Dict[str, Any]:
        """
        Align multiple data arrays to the same length and validate.
        
        Args:
            *arrays: Variable number of input arrays
            
        Returns:
            Dictionary with aligned data and metadata
        """
        # Find minimum length across all arrays
        min_length = min(len(arr) for arr in arrays)
        
        if min_length == 0:
            raise ValueError("All input arrays must have at least one element")
        
        # Check for data quality issues
        for i, arr in enumerate(arrays):
            if np.isnan(arr[:min_length]).any():
                warnings.warn(f"Array {i} contains NaN values. Results may be affected.")
        
        return {
            'heart_rate': np.array(arrays[0][:min_length]),
            'deviation': np.array(arrays[1][:min_length]),
            'rate_of_change': np.array(arrays[2][:min_length]),
            'pupil_diameter': np.array(arrays[3][:min_length]),
            'time': np.array(arrays[4][:min_length]),
            'n_samples': min_length
        }
    
    def _calculate_thresholds(self, pupil_diameter: np.ndarray, 
                               hr_threshold: float,
                               min_time_jasat: Optional[float],
                               use_dynamic_thresholds: bool) -> Dict[str, Any]:
        """
        Calculate all thresholds for state classification.
        
        Args:
            pupil_diameter: Pupil diameter measurements
            hr_threshold: Heart rate threshold
            min_time_jasat: JASAT time for dynamic thresholds
            use_dynamic_thresholds: Whether to use dynamic thresholds
            
        Returns:
            Dictionary containing all thresholds
        """
        # Calculate pupil diameter threshold
        pd_mean = np.nanmean(pupil_diameter)
        pd_std = np.nanstd(pupil_diameter)
        pd_threshold = pd_mean + pd_std
        
        # Set deviation and rate thresholds
        if use_dynamic_thresholds and min_time_jasat is not None:
            dev_before = self.DEV_THRESHOLD_BEFORE_JASAT
            dev_after = self.DEV_THRESHOLD_AFTER_JASAT
            rate_before = self.RATE_THRESHOLD_BEFORE_JASAT
            rate_after = self.RATE_THRESHOLD_AFTER_JASAT
        else:
            dev_before = self.DEFAULT_DEV_THRESHOLD
            dev_after = self.DEFAULT_DEV_THRESHOLD
            rate_before = self.DEFAULT_RATE_THRESHOLD
            rate_after = self.DEFAULT_RATE_THRESHOLD
        
        return {
            'hr_threshold': hr_threshold,
            'pd_threshold': pd_threshold,
            'pd_mean': pd_mean,
            'pd_std': pd_std,
            'dev_before_jasat': dev_before,
            'dev_after_jasat': dev_after,
            'rate_before_jasat': rate_before,
            'rate_after_jasat': rate_after,
            'min_time_jasat': min_time_jasat,
            'dynamic_thresholds': use_dynamic_thresholds and min_time_jasat is not None
        }
    
    def _classify_states(self, heart_rate: np.ndarray, deviation: np.ndarray,
                          rate_of_change: np.ndarray, pupil_diameter: np.ndarray,
                          time: np.ndarray, thresholds: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Classify performance states for each time point.
        
        Args:
            heart_rate: Heart rate data
            deviation: Deviation data
            rate_of_change: Rate of change data
            pupil_diameter: Pupil diameter data
            time: Time data
            thresholds: Threshold dictionary
            
        Returns:
            Tuple of (performance_states, hr_pd_states)
        """
        performance_states = []
        hr_pd_states = []
        
        for i in range(len(heart_rate)):
            # Determine which thresholds to use based on time
            current_time = time[i]
            if thresholds['dynamic_thresholds'] and current_time >= thresholds['min_time_jasat']:
                current_dev_threshold = thresholds['dev_after_jasat']
                current_rate_threshold = thresholds['rate_after_jasat']
            else:
                current_dev_threshold = thresholds['dev_before_jasat']
                current_rate_threshold = thresholds['rate_before_jasat']
            
            # Flight control classification
            is_controlled = (deviation[i] <= current_dev_threshold) and \
                           (rate_of_change[i] <= current_rate_threshold)
            
            # Physiological state classification
            is_high_hr = heart_rate[i] > thresholds['hr_threshold']
            is_high_pd = pupil_diameter[i] > thresholds['pd_threshold']
            
            # HR/PD based stress state
            if is_high_hr and is_high_pd:
                hr_pd_state = "Stressed"
            else:
                hr_pd_state = "Relaxed"
            
            hr_pd_states.append(hr_pd_state)
            
            # Combined state
            if hr_pd_state == "Stressed" and not is_controlled:
                performance_states.append("Stressed Uncontrolled")
            elif hr_pd_state == "Stressed" and is_controlled:
                performance_states.append("Stressed Controlled")
            elif hr_pd_state == "Relaxed" and not is_controlled:
                performance_states.append("Relaxed Uncontrolled")
            else:
                performance_states.append("Relaxed Controlled")
        
        return performance_states, hr_pd_states
    
    def _calculate_statistics(self, performance_states: List[str],
                               hr_pd_states: List[str], time: np.ndarray,
                               thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from state classifications.
        
        Args:
            performance_states: List of performance states
            hr_pd_states: List of physiological states
            time: Time array
            thresholds: Threshold dictionary
            
        Returns:
            Dictionary containing all statistics
        """
        n_samples = len(performance_states)
        total_time = n_samples / self.sampling_freq
        
        # Calculate state counts and percentages
        state_counts = {}
        hr_pd_counts = {
            "Stressed": hr_pd_states.count("Stressed"),
            "Relaxed": hr_pd_states.count("Relaxed")
        }
        
        for state in set(performance_states):
            state_counts[state] = performance_states.count(state)
        
        # Calculate state durations
        state_durations = self._calculate_state_durations(performance_states)
        
        # Calculate transitions
        transitions = self._calculate_transitions(performance_states, time)
        
        # Calculate time-based statistics
        time_stats = self._calculate_time_statistics(time, thresholds)
        
        return {
            'n_samples': n_samples,
            'total_time': total_time,
            'state_counts': state_counts,
            'hr_pd_counts': hr_pd_counts,
            'state_durations': state_durations,
            'transitions': transitions,
            'time_statistics': time_stats
        }
    
    def _calculate_state_durations(self, states: List[str]) -> Dict[str, Any]:
        """
        Calculate duration statistics for each state.
        
        Args:
            states: List of state labels
            
        Returns:
            Dictionary with duration statistics per state
        """
        durations = {}
        states_array = np.array(states)
        
        for state in set(states):
            state_mask = states_array == state
            
            if not np.any(state_mask):
                durations[state] = self._empty_duration_stats()
                continue
            
            # Find consecutive runs
            diff = np.diff(np.concatenate(([0], state_mask.astype(int), [0])))
            run_starts = np.where(diff == 1)[0]
            run_ends = np.where(diff == -1)[0]
            durations_samples = run_ends - run_starts
            
            # Convert to seconds
            durations_seconds = durations_samples / self.sampling_freq
            
            durations[state] = {
                'num_episodes': len(durations_samples),
                'durations_samples': durations_samples,
                'durations_seconds': durations_seconds,
                'total_duration_seconds': np.sum(durations_seconds),
                'avg_duration_seconds': np.mean(durations_seconds) if len(durations_seconds) > 0 else 0,
                'max_duration_seconds': np.max(durations_seconds) if len(durations_seconds) > 0 else 0,
                'min_duration_seconds': np.min(durations_seconds) if len(durations_seconds) > 0 else 0,
                'avg_duration_samples': np.mean(durations_samples) if len(durations_samples) > 0 else 0,
                'max_duration_samples': np.max(durations_samples) if len(durations_samples) > 0 else 0,
                'min_duration_samples': np.min(durations_samples) if len(durations_samples) > 0 else 0
            }
        
        return durations
    
    def _empty_duration_stats(self) -> Dict[str, Any]:
        """Return empty duration statistics dictionary."""
        return {
            'num_episodes': 0,
            'durations_samples': np.array([]),
            'durations_seconds': np.array([]),
            'total_duration_seconds': 0,
            'avg_duration_seconds': 0,
            'max_duration_seconds': 0,
            'min_duration_seconds': 0,
            'avg_duration_samples': 0,
            'max_duration_samples': 0,
            'min_duration_samples': 0
        }
    
    def _calculate_transitions(self, states: List[str], 
                                time: np.ndarray) -> Dict[str, Any]:
        """
        Calculate state transitions including self-transitions.
        
        Args:
            states: List of state labels
            time: Time array
            
        Returns:
            Dictionary with transition statistics
        """
        transitions = []
        state_changes = []
        
        for i in range(1, len(states)):
            transition = {
                'time': time[i],
                'from_state': states[i-1],
                'to_state': states[i],
                'index': i,
                'is_state_change': states[i] != states[i-1]
            }
            transitions.append(transition)
            
            if states[i] != states[i-1]:
                state_changes.append(transition)
        
        # Count transition types
        transition_counts = {}
        state_change_counts = {}
        
        for t in transitions:
            key = f"{t['from_state']} → {t['to_state']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
            
            if t['is_state_change']:
                state_change_counts[key] = state_change_counts.get(key, 0) + 1
        
        return {
            'all_transitions': transitions,
            'state_changes': state_changes,
            'total_transitions': len(transitions),
            'total_state_changes': len(state_changes),
            'transition_counts': transition_counts,
            'state_change_counts': state_change_counts
        }
    
    def _calculate_time_statistics(self, time: np.ndarray,
                                    thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate time-based statistics.
        
        Args:
            time: Time array
            thresholds: Threshold dictionary
            
        Returns:
            Dictionary with time statistics
        """
        total_time = time[-1] - time[0]
        
        if thresholds['dynamic_thresholds']:
            jasat_time = thresholds['min_time_jasat']
            time_before_jasat = np.sum(time < jasat_time) / self.sampling_freq
            time_after_jasat = total_time - time_before_jasat
            
            return {
                'total_time': total_time,
                'time_before_jasat': time_before_jasat,
                'time_after_jasat': time_after_jasat,
                'pct_before_jasat': (time_before_jasat / total_time) * 100,
                'pct_after_jasat': (time_after_jasat / total_time) * 100
            }
        else:
            return {'total_time': total_time}
    
    def _print_analysis_header(self, n_samples: int, dataset_name: str):
        """Print analysis header information."""
        print("\n" + "="*80)
        print("PERFORMANCE STATE ANALYSIS")
        print("="*80)
        print(f"Analyzing {n_samples} data points at {self.sampling_freq} Hz")
        if dataset_name:
            print(f"Dataset: {dataset_name}")
        print(f"Total recording time: {n_samples/self.sampling_freq:.2f} seconds")
    
    def _print_threshold_summary(self, thresholds: Dict[str, Any], 
                                  hr_threshold: float):
        """Print threshold summary."""
        print("\n" + "-"*80)
        print("THRESHOLD SUMMARY")
        print("-"*80)
        
        print(f"Heart Rate Threshold: {hr_threshold:.1f} bpm")
        print(f"Pupil Diameter Threshold: {thresholds['pd_threshold']:.2f} mm")
        print(f"  - Mean: {thresholds['pd_mean']:.2f} mm")
        print(f"  - Std Dev: {thresholds['pd_std']:.2f} mm")
        
        if thresholds['dynamic_thresholds']:
            print("\nDynamic Thresholds (based on JASAT time):")
            print(f"  Before JASAT (<{thresholds['min_time_jasat']:.1f}s):")
            print(f"    - Deviation: {thresholds['dev_before_jasat']:.1f} ft")
            print(f"    - Rate of Change: {thresholds['rate_before_jasat']:.1f} ft/s")
            print(f"  After JASAT (≥{thresholds['min_time_jasat']:.1f}s):")
            print(f"    - Deviation: {thresholds['dev_after_jasat']:.1f} ft")
            print(f"    - Rate of Change: {thresholds['rate_after_jasat']:.1f} ft/s")
        else:
            print("\nConstant Thresholds:")
            print(f"  Deviation: {thresholds['dev_before_jasat']:.1f} ft")
            print(f"  Rate of Change: {thresholds['rate_before_jasat']:.1f} ft/s")
    
    def _plot_results(self, time: np.ndarray, heart_rate: np.ndarray,
                       deviation: np.ndarray, rate_of_change: np.ndarray,
                       pupil_diameter: np.ndarray, performance_states: List[str],
                       thresholds: Dict[str, Any]):
        """
        Generate visualization plots for the analysis results.
        
        Args:
            time: Time array
            heart_rate: Heart rate data
            deviation: Deviation data
            rate_of_change: Rate of change data
            pupil_diameter: Pupil diameter data
            performance_states: List of performance states
            thresholds: Threshold dictionary
        """
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        fig.suptitle('Performance State Analysis Results', fontsize=16)
        
        # Plot 1: Heart Rate
        self._plot_metric_with_states(
            axes[0], time, heart_rate, performance_states,
            "Heart Rate (bpm)", thresholds['hr_threshold'],
            "HR Threshold"
        )
        
        # Plot 2: Pupil Diameter
        self._plot_metric_with_states(
            axes[1], time, pupil_diameter, performance_states,
            "Pupil Diameter (mm)", thresholds['pd_threshold'],
            "PD Threshold", secondary_line=thresholds['pd_mean'],
            secondary_label="PD Mean"
        )
        
        # Plot 3: Deviation
        self._plot_deviation_rate(
            axes[2], time, deviation, performance_states,
            thresholds, is_deviation=True
        )
        
        # Plot 4: Rate of Change
        self._plot_deviation_rate(
            axes[3], time, rate_of_change, performance_states,
            thresholds, is_deviation=False
        )
        
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    
    def _plot_metric_with_states(self, ax, time, data, states,
                                   ylabel, threshold, threshold_label,
                                   secondary_line=None, secondary_label=None):
        """
        Plot a metric with state-based coloring.
        
        Args:
            ax: Matplotlib axis
            time: Time array
            data: Data array to plot
            states: State labels
            ylabel: Y-axis label
            threshold: Threshold value
            threshold_label: Label for threshold line
            secondary_line: Optional secondary line value
            secondary_label: Label for secondary line
        """
        # Create state abbreviations
        state_abbr = {
            "Stressed Uncontrolled": "SU",
            "Relaxed Uncontrolled": "RU",
            "Stressed Controlled": "SC",
            "Relaxed Controlled": "RC"
        }
        
        # Plot each state with its color
        for state, color in self.STATE_COLORS.items():
            mask = np.array(states) == state
            if np.any(mask):
                ax.scatter(time[mask], data[mask], 
                          color=color, label=state_abbr.get(state, state),
                          alpha=0.6, s=5)
        
        # Plot threshold line
        ax.axhline(y=threshold, color='black', linestyle='--', 
                  alpha=0.8, linewidth=1.5, label=threshold_label)
        
        # Plot secondary line if provided
        if secondary_line is not None:
            ax.axhline(y=secondary_line, color='gray', linestyle=':', 
                      alpha=0.7, linewidth=1, label=secondary_label)
        
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2)
    
    def _plot_deviation_rate(self, ax, time, data, states,
                               thresholds, is_deviation=True):
        """
        Plot deviation or rate of change with dynamic thresholds.
        
        Args:
            ax: Matplotlib axis
            time: Time array
            data: Data array
            states: State labels
            thresholds: Threshold dictionary
            is_deviation: True for deviation, False for rate of change
        """
        # Set labels and thresholds based on type
        if is_deviation:
            ylabel = "FPD (ft)"
            before_threshold = thresholds['dev_before_jasat']
            after_threshold = thresholds['dev_after_jasat']
            before_label = f"Dev Before JASAT: {before_threshold:.1f} ft"
            after_label = f"Dev After JASAT: {after_threshold:.1f} ft"
        else:
            ylabel = "ROCD (ft/s)"
            before_threshold = thresholds['rate_before_jasat']
            after_threshold = thresholds['rate_after_jasat']
            before_label = f"Rate Before JASAT: {before_threshold:.1f} ft/s"
            after_label = f"Rate After JASAT: {after_threshold:.1f} ft/s"
        
        # Plot data with state colors
        for state, color in self.STATE_COLORS.items():
            mask = np.array(states) == state
            if np.any(mask):
                state_abbr = state[:2].upper() if len(state) > 2 else state
                ax.scatter(time[mask], data[mask], 
                          color=color, label=state_abbr,
                          alpha=0.6, s=5)
        
        # Plot dynamic thresholds
        if thresholds['dynamic_thresholds']:
            # Vertical line at JASAT
            ax.axvline(x=thresholds['min_time_jasat'], color='black', 
                      linestyle=':', alpha=0.5, linewidth=1,
                      label=f"JASAT: {thresholds['min_time_jasat']:.1f}s")
            
            # Before JASAT threshold
            x_before = [time[0], thresholds['min_time_jasat']]
            ax.plot(x_before, [before_threshold, before_threshold],
                   color='black', linestyle='--', alpha=0.8,
                   linewidth=1.5, label=before_label)
            
            # After JASAT threshold
            x_after = [thresholds['min_time_jasat'], time[-1]]
            ax.plot(x_after, [after_threshold, after_threshold],
                   color='black', linestyle='--', alpha=0.8,
                   linewidth=1.5, label=after_label)
        else:
            # Single threshold line
            ax.axhline(y=before_threshold, color='black', linestyle='--',
                      alpha=0.8, linewidth=1.5,
                      label=f"Threshold: {before_threshold:.1f} {'ft' if is_deviation else 'ft/s'}")
        
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', ncol=2)
    
    def _compile_results(self, performance_states: List[str],
                          hr_pd_states: List[str], statistics: Dict[str, Any],
                          thresholds: Dict[str, Any],
                          heart_rate: np.ndarray, deviation: np.ndarray,
                          rate_of_change: np.ndarray, pupil_diameter: np.ndarray,
                          time: np.ndarray) -> Dict[str, Any]:
        """
        Compile all results into a single dictionary.
        
        Args:
            performance_states: List of performance states
            hr_pd_states: List of physiological states
            statistics: Statistics dictionary
            thresholds: Threshold dictionary
            heart_rate: Heart rate data
            deviation: Deviation data
            rate_of_change: Rate of change data
            pupil_diameter: Pupil diameter data
            time: Time data
            
        Returns:
            Comprehensive results dictionary
        """
        return {
            'performance_states': performance_states,
            'hr_pd_states': hr_pd_states,
            'statistics': statistics,
            'thresholds': thresholds,
            'data': {
                'heart_rate': heart_rate,
                'deviation': deviation,
                'rate_of_change': rate_of_change,
                'pupil_diameter': pupil_diameter,
                'time': time
            },
            'metadata': {
                'sampling_freq': self.sampling_freq,
                'analyzer_version': '1.0.0'
            }
        }


def analyze_performance_states(heart_rate_data, deviation_data, 
                                rate_of_change_data, pupil_data,
                                time_data, hr_threshold,
                                min_time_jasat=None, dataset_name="",
                                sampling_freq=60.0):
    """
    Convenience function for performance state analysis.
    
    This is a wrapper around PerformanceStateAnalyzer for backward compatibility.
    
    Args:
        heart_rate_data: Heart rate measurements (bpm)
        deviation_data: Flight path deviation measurements (ft)
        rate_of_change_data: Rate of change of deviation (ft/s)
        pupil_data: Pupil diameter measurements (mm)
        time_data: Time stamps (seconds)
        hr_threshold: Heart rate threshold for stress classification (bpm)
        min_time_jasat: JASAT time for dynamic thresholds (seconds)
        dataset_name: Name of the dataset for reporting
        sampling_freq: Sampling frequency in Hz (default: 60 Hz)
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    analyzer = PerformanceStateAnalyzer(sampling_freq=sampling_freq)
    
    return analyzer.analyze(
        heart_rate_data=heart_rate_data,
        deviation_data=deviation_data,
        rate_of_change_data=rate_of_change_data,
        pupil_data=pupil_data,
        time_data=time_data,
        hr_threshold=hr_threshold,
        min_time_jasat=min_time_jasat,
        dataset_name=dataset_name
    )


if __name__ == "__main__":
    # Example usage
    print("Human Behavior and Task Performance Modeling for AI Perception")
    print("="*60)
    print("\nThis module provides tools for analyzing pilot performance states.")
    print("For usage examples, see the examples/ directory.")
