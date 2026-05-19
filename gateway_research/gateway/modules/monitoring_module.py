"""
Monitoring Service Module for Fallyx Gateway

This module handles system monitoring, health checks, and resource management.
"""

import logging
import time
import threading
import os
import json
import subprocess
import psutil
import shutil

logger = logging.getLogger("MonitoringService")

class MonitoringService:
    """
    Monitoring Service for system health and resource management.
    
    This module is responsible for:
    - Monitoring system resources (CPU, memory, storage)
    - Performing periodic health checks
    - Adjusting resource allocation based on system load
    - Logging system metrics
    """
    
    def __init__(self):
        """Initialize the Monitoring Service."""
        logger.info("Initializing Monitoring Service")
        self.running = False
        self.check_interval = 60  # seconds
        self.last_check_time = 0
        self.system_metrics = {}
        self.lock = threading.Lock()  # For thread-safe operations
    
    def run(self):
        """Run the Monitoring Service main loop."""
        logger.info("Starting Monitoring Service")
        self.running = True
        
        while self.running:
            # Check if it's time to perform a health check
            current_time = time.time()
            if current_time - self.last_check_time >= self.check_interval:
                self._perform_health_check()
                self.last_check_time = current_time
            
            logger.debug("Monitoring Service running...")
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the Monitoring Service."""
        logger.info("Stopping Monitoring Service")
        self.running = False
    
    def _perform_health_check(self):
        """Perform a system health check."""
        logger.info("Performing system health check")
        
        # Collect system metrics
        self._collect_system_metrics()
        
        # Log system metrics
        self._log_system_metrics()
        
        # Check for resource constraints
        self._check_resource_constraints()
    
    def _collect_system_metrics(self):
        """Collect system metrics (CPU, memory, storage, etc.)."""
        logger.debug("Collecting system metrics")
        
        with self.lock:
            self.system_metrics = {
                "timestamp": int(time.time()),
                "cpu_usage": self._get_cpu_usage(),
                "cpu_temperature": self._get_cpu_temperature(),
                "memory_usage": self._get_memory_usage(),
                "storage": self._get_storage_usage(),
                "network": self._get_network_info(),
                "uptime": self._get_uptime(),
                "load_average": self._get_load_average()
            }
    
    def _get_cpu_usage(self):
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0
    
    def _get_cpu_temperature(self):
        """Get CPU temperature from thermal zone (Raspberry Pi specific)."""
        try:
            # Try multiple thermal zone paths
            thermal_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/devices/virtual/thermal/thermal_zone0/temp"
            ]
            
            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        return temp_millidegrees / 1000.0
            
            # Fallback: try vcgencmd for Raspberry Pi
            result = subprocess.run(['vcgencmd', 'measure_temp'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                # Extract temperature from "temp=XX.X'C"
                temp = float(temp_str.split('=')[1].split("'")[0])
                return temp
                
        except Exception as e:
            logger.debug(f"Could not get CPU temperature: {e}")
        
        return None
    
    def _get_memory_usage(self):
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total": round(memory.total / (1024**2)),  # MB
                "used": round(memory.used / (1024**2)),   # MB
                "free": round(memory.available / (1024**2)),  # MB
                "percent": memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {"total": 0, "used": 0, "free": 0, "percent": 0}
    
    def _get_storage_usage(self):
        """Get storage usage for root filesystem."""
        try:
            usage = shutil.disk_usage('/')
            total = usage.total
            used = usage.used
            free = usage.free
            
            return {
                "total": round(total / (1024**3), 2),  # GB
                "used": round(used / (1024**3), 2),   # GB
                "free": round(free / (1024**3), 2),   # GB
                "percent": round((used / total) * 100, 1)
            }
        except Exception as e:
            logger.error(f"Error getting storage usage: {e}")
            return {"total": 0, "used": 0, "free": 0, "percent": 0}
    
    def _get_network_info(self):
        """Get network information."""
        network_info = {
            "wifi_signal": None,
            "ip_address": "0.0.0.0",
            "interface": None
        }
        
        try:
            # Get IP address of active interface
            addrs = psutil.net_if_addrs()
            for interface, addr_list in addrs.items():
                if interface.startswith(('wlan', 'eth', 'enp')):
                    for addr in addr_list:
                        if addr.family == 2:  # AF_INET (IPv4)
                            if not addr.address.startswith('127.'):
                                network_info["ip_address"] = addr.address
                                network_info["interface"] = interface
                                break
            
            # Get WiFi signal strength for wireless interfaces
            if network_info["interface"] and network_info["interface"].startswith('wlan'):
                try:
                    result = subprocess.run(['iwconfig', network_info["interface"]],
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        output = result.stdout
                        # Parse signal strength from iwconfig output
                        for line in output.split('\n'):
                            if 'Signal level' in line:
                                # Extract signal level (e.g., "Signal level=-45 dBm")
                                signal_part = line.split('Signal level=')[1].split()[0]
                                network_info["wifi_signal"] = int(signal_part)
                                break
                except Exception as e:
                    logger.debug(f"Could not get WiFi signal strength: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
        
        return network_info
    
    def _get_uptime(self):
        """Get system uptime in seconds."""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
                return int(uptime_seconds)
        except Exception as e:
            logger.error(f"Error getting uptime: {e}")
            return 0
    
    def _get_load_average(self):
        """Get system load average."""
        try:
            load1, load5, load15 = os.getloadavg()
            return {
                "1min": round(load1, 2),
                "5min": round(load5, 2),
                "15min": round(load15, 2)
            }
        except Exception as e:
            logger.error(f"Error getting load average: {e}")
            return {"1min": 0, "5min": 0, "15min": 0}
    
    def _log_system_metrics(self):
        """Log system metrics for monitoring."""
        cpu_usage = self.system_metrics.get('cpu_usage', 0)
        cpu_temp = self.system_metrics.get('cpu_temperature')
        memory = self.system_metrics.get('memory_usage', {})
        storage = self.system_metrics.get('storage', {})
        network = self.system_metrics.get('network', {})
        load_avg = self.system_metrics.get('load_average', {})
        
        temp_str = f", Temp: {cpu_temp:.1f}°C" if cpu_temp else ""
        
        logger.info(f"System metrics - CPU: {cpu_usage:.1f}%{temp_str}, "
                   f"Memory: {memory.get('percent', 0):.1f}% ({memory.get('used', 0)}MB), "
                   f"Storage: {storage.get('percent', 0):.1f}%, "
                   f"Load: {load_avg.get('1min', 0)}, "
                   f"IP: {network.get('ip_address', 'N/A')}")
    
    def _check_resource_constraints(self):
        """Check for resource constraints and adjust if necessary."""
        logger.debug("Checking resource constraints")
        
        cpu_usage = self.system_metrics.get('cpu_usage', 0)
        memory_percent = self.system_metrics.get('memory_usage', {}).get('percent', 0)
        storage_percent = self.system_metrics.get('storage', {}).get('percent', 0)
        cpu_temp = self.system_metrics.get('cpu_temperature')
        
        # Check CPU usage
        if cpu_usage > 80:
            logger.warning(f"High CPU usage detected: {cpu_usage:.1f}%")
        
        # Check memory usage
        if memory_percent > 85:
            logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        
        # Check storage usage
        if storage_percent > 90:
            logger.warning(f"High storage usage detected: {storage_percent:.1f}%")
        
        # Check CPU temperature (if available)
        if cpu_temp and cpu_temp > 70:
            logger.warning(f"High CPU temperature detected: {cpu_temp:.1f}°C")
            if cpu_temp > 80:
                logger.error(f"Critical CPU temperature: {cpu_temp:.1f}°C")
    
    def get_system_status(self):
        """Get the current system status."""
        with self.lock:
            return self.system_metrics.copy()
    
    def log_event(self, event_type, event_data):
        """Log a system event."""
        logger.info(f"System event: {event_type} - {event_data}")
        # This would log events to a file or database for later analysis
