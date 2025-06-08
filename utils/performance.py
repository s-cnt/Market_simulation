import time
import numpy as np
import logging
from collections import deque

class PerformanceMonitor:
    """
    Monitor and track performance metrics for the trading simulator.
    """
    
    def __init__(self, window_size=100):
        self.logger = logging.getLogger("performance_monitor")
        self.window_size = window_size
        
        # Tracking latency metrics with rolling windows
        self.processing_latencies = deque(maxlen=window_size)
        self.ui_update_latencies = deque(maxlen=window_size)
        self.end_to_end_latencies = deque(maxlen=window_size)
        
        # Tracking tick rates
        self.last_tick_time = None
        self.tick_intervals = deque(maxlen=window_size)
        
    def record_processing_latency(self, latency_ms):
        """Record time taken to process a market data tick."""
        self.processing_latencies.append(latency_ms)
        
    def record_ui_update_latency(self, latency_ms):
        """Record time taken to update the UI."""
        self.ui_update_latencies.append(latency_ms)
        
    def record_end_to_end_latency(self, latency_ms):
        """Record total time from receiving data to completing UI update."""
        self.end_to_end_latencies.append(latency_ms)
        
    def record_tick(self):
        """Record the arrival of a new market data tick."""
        current_time = time.time()
        if self.last_tick_time is not None:
            interval = current_time - self.last_tick_time
            self.tick_intervals.append(interval)
        self.last_tick_time = current_time
        
    def get_processing_latency_stats(self):
        """Get statistics about processing latency."""
        if not self.processing_latencies:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
            
        latencies = np.array(self.processing_latencies)
        return {
            "avg": np.mean(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
    def get_ui_update_latency_stats(self):
        """Get statistics about UI update latency."""
        if not self.ui_update_latencies:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
            
        latencies = np.array(self.ui_update_latencies)
        return {
            "avg": np.mean(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
    def get_end_to_end_latency_stats(self):
        """Get statistics about end-to-end latency."""
        if not self.end_to_end_latencies:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
            
        latencies = np.array(self.end_to_end_latencies)
        return {
            "avg": np.mean(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
    def get_tick_rate(self):
        """Calculate the average number of ticks per second."""
        if not self.tick_intervals:
            return 0
            
        avg_interval = np.mean(self.tick_intervals)
        return 1.0 / avg_interval if avg_interval > 0 else 0
        
    def get_current_latency(self):
        """Get the most recent processing latency."""
        if not self.processing_latencies:
            return 0
        return self.processing_latencies[-1]