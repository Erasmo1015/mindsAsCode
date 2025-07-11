#!/usr/bin/env python3
"""
Memory monitoring script for FSM evaluation.
Run this in a separate terminal to monitor GPU memory usage.
"""

import subprocess
import time
import psutil
import os

def get_gpu_memory():
    """Get GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            memory_info = []
            for line in lines:
                used, total = map(int, line.split(', '))
                memory_info.append({'used': used, 'total': total, 'percent': (used/total)*100})
            return memory_info
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
    return None

def get_system_memory():
    """Get system memory usage."""
    memory = psutil.virtual_memory()
    return {
        'used': memory.used // (1024**3),  # GB
        'total': memory.total // (1024**3),  # GB
        'percent': memory.percent
    }

def monitor_memory(interval=5):
    """Monitor memory usage continuously."""
    print("Memory Monitor Started (Press Ctrl+C to stop)")
    print("=" * 60)
    print(f"{'Time':<12} {'GPU Used':<10} {'GPU Total':<10} {'GPU %':<8} {'RAM Used':<10} {'RAM Total':<10} {'RAM %':<8}")
    print("=" * 60)
    
    try:
        while True:
            timestamp = time.strftime("%H:%M:%S")
            
            # Get GPU memory
            gpu_memory = get_gpu_memory()
            if gpu_memory:
                gpu_used = f"{gpu_memory[0]['used']}MB"
                gpu_total = f"{gpu_memory[0]['total']}MB"
                gpu_percent = f"{gpu_memory[0]['percent']:.1f}%"
            else:
                gpu_used = gpu_total = gpu_percent = "N/A"
            
            # Get system memory
            ram_memory = get_system_memory()
            ram_used = f"{ram_memory['used']}GB"
            ram_total = f"{ram_memory['total']}GB"
            ram_percent = f"{ram_memory['percent']:.1f}%"
            
            print(f"{timestamp:<12} {gpu_used:<10} {gpu_total:<10} {gpu_percent:<8} {ram_used:<10} {ram_total:<10} {ram_percent:<8}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMemory monitoring stopped.")

if __name__ == "__main__":
    monitor_memory() 