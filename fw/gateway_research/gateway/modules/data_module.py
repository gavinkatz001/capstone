"""
Data Manager Module for Fallyx Gateway

This module handles processing and formatting sensor data,
validating data integrity, and managing temporary storage.
"""

import logging
import time
import threading
import json
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("DataManager")

class DataManager:
    """
    Data Manager for processing and formatting sensor data.
    
    This module is responsible for:
    - Creating structured JSON from sensor data
    - Validating data integrity
    - Managing temporary storage of data
    - Uploading incomplete packet sets when timeouts occur (preserves partial data)
    """
    
    def __init__(self, gateway_id="gateway-location-123"):
        """Initialize the Data Manager module."""
        logger.info("Initializing Data Manager")
        self.network_module = None
        self.running = False
        self.gateway_id = gateway_id
        self.lock = threading.Lock()  # For thread-safe operations
        
        # Packet buffering for aggregation
        self.packet_buffers = defaultdict(dict)  # {mac_address: {packet_type: packet_data}}
        self.packet_timestamps = defaultdict(dict)  # {mac_address: {packet_type: timestamp}}
        self.expected_packet_types = {0xA1, 0x41, 0x45}  # Expected packet types
        self.packet_timeout = 45  # Timeout in seconds for incomplete packet sets
    
    def set_network_module(self, network_module):
        """Set the network module reference for sending data."""
        self.network_module = network_module
        logger.info("Network module set")
    
    def run(self):
        """Run the Data Manager main loop."""
        logger.info("Starting Data Manager")
        self.running = True
        
        while self.running:
            logger.debug("Data Manager running...")
            self._cleanup_expired_packets()
            time.sleep(5)  # Sleep to prevent CPU hogging
    
    def stop(self):
        """Stop the Data Manager."""
        logger.info("Stopping Data Manager")
        self.running = False
    
    def synthesize_timestamps(self, sample_count):
        """
        Synthesize timestamps for sensor data when timestamps are missing or invalid.
        
        Args:
            sample_count: Number of samples that need timestamps
            
        Returns:
            List of synthesized timestamps (Unix timestamps in seconds)
        """
        if sample_count <= 0:
            return []
            
        # Get current time and subtract 1/2 minute (data is from 1/2 minute ago)
        end_time = datetime.now() - timedelta(seconds=30)
        
        # Calculate the time interval between samples
        # For 1/2 minute of data, distribute samples evenly across 30 seconds
        interval_seconds = 30.0 / sample_count
        
        timestamps = []
        for i in range(sample_count):
            # Calculate timestamp for this sample
            sample_time = end_time - timedelta(seconds=(sample_count - 1 - i) * interval_seconds)
            # Convert to Unix timestamp
            unix_timestamp = int(sample_time.timestamp())
            timestamps.append(unix_timestamp)
            
        logger.info(f"Synthesized {sample_count} timestamps with {interval_seconds:.2f}s intervals")
        return timestamps
    
    def _synthesize_sensor_timestamps_from_pressure(self, sample_count, last_pressure_ts_ms,
                                                   sensor_rate_hz, downsample_factor, sensor_type="sensor"):
        """
        Synthesize sensor timestamps based on pressure timestamp and ODR settings.
        
        Args:
            sample_count: Number of sensor samples
            last_pressure_ts_ms: Last pressure timestamp in milliseconds (sensor uptime)
            sensor_rate_hz: Mapped sensor ODR in Hz
            downsample_factor: Mapped downsample factor
            sensor_type: Type of sensor for logging (e.g., "accelerometer", "gyroscope")
        
        Returns:
            List of timestamps in milliseconds (sensor uptime)
        """
        if sample_count <= 0 or not sensor_rate_hz or not downsample_factor:
            logger.warning(f"Invalid parameters for {sensor_type} timestamp synthesis: "
                          f"samples={sample_count}, rate={sensor_rate_hz}, downsample={downsample_factor}")
            return []
        
        # Calculate effective sampling rate after downsampling
        effective_hz = sensor_rate_hz / downsample_factor
        interval_ms = 1000.0 / effective_hz
        
        logger.info(f"Synthesizing {sensor_type} timestamps: {sensor_rate_hz}Hz ODR, "
                   f"{downsample_factor}x downsample = {effective_hz}Hz effective rate")
        
        # Generate timestamps working backward from last pressure timestamp
        timestamps = []
        for i in range(sample_count):
            # Calculate timestamp for this sample (working backward)
            sample_offset = (sample_count - 1 - i) * interval_ms
            timestamp = last_pressure_ts_ms - sample_offset
            timestamps.append(int(timestamp))
        
        logger.info(f"Generated {len(timestamps)} {sensor_type} timestamps from "
                   f"{timestamps[0] if timestamps else 'N/A'} to {timestamps[-1] if timestamps else 'N/A'} ms")
        return timestamps
    
    def _validate_pressure_timing_data(self, pressure_data, gpio_data, sensor_type):
        """Validate that we have sufficient data for pressure-based timing."""
        if not pressure_data or not gpio_data:
            return False
            
        pressure_ts = pressure_data.get('pressureTs', [])
        if not pressure_ts or len(pressure_ts) == 0:
            return False
            
        if sensor_type == "accelerometer":
            rate = gpio_data.get('AccelRate')
            downsample = gpio_data.get('AccelDownsample')
        elif sensor_type == "gyroscope":
            rate = gpio_data.get('GyroRate')
            downsample = gpio_data.get('GyroDownsample')
        else:
            return False
            
        return (rate is not None and downsample is not None and
                rate > 0 and downsample > 0)
    
    def _cleanup_expired_packets(self):
        """Process and upload expired packet buffers that haven't received all expected packets."""
        current_time = time.time()
        expired_macs = []
        
        with self.lock:
            for mac_address, timestamps in self.packet_timestamps.items():
                # Check if any packet in this MAC's buffer has expired
                for packet_type, timestamp in timestamps.items():
                    if current_time - timestamp > self.packet_timeout:
                        expired_macs.append(mac_address)
                        break
        
        # Process expired MAC addresses - upload incomplete data instead of discarding
        for mac_address in expired_macs:
            logger.warning(f"Timeout reached for sensor {mac_address} - uploading incomplete packet set")
            self._upload_incomplete_packets(mac_address)
    
    def _upload_incomplete_packets(self, mac_address):
        """Upload incomplete packet data when timeout is reached."""
        with self.lock:
            if mac_address not in self.packet_buffers:
                logger.error(f"No packet buffer found for MAC {mac_address}")
                return
            
            packets = self.packet_buffers[mac_address].copy()
            timestamps = self.packet_timestamps[mac_address].copy()
            
            # Clear the buffers for this MAC
            del self.packet_buffers[mac_address]
            del self.packet_timestamps[mac_address]
        
        # Log what packets we have and what's missing
        received_types = set(packets.keys())
        missing_types = self.expected_packet_types - received_types
        logger.info(f"Uploading incomplete data for sensor {mac_address}")
        logger.info(f"  Available packets: {[hex(t) for t in received_types]}")
        logger.info(f"  Missing packets: {[hex(t) for t in missing_types]}")
        
        # Calculate how long each packet has been waiting
        current_time = time.time()
        for packet_type, timestamp in timestamps.items():
            wait_time = current_time - timestamp
            logger.info(f"  Packet {hex(packet_type)} waited {wait_time:.1f} seconds")
        
        # Initialize the structured telemetry format with incomplete data metadata
        telemetry_data = {
            "Location": self.gateway_id,
            "sensor_mac": mac_address,
            "data_complete": False,  # Flag indicating incomplete data
            "missing_packet_types": [hex(t) for t in missing_types],
            "timeout_reason": "packet_set_timeout",
            "packet_timeout_seconds": self.packet_timeout,
        }
        
        # Process available packets (same logic as complete packets)
        # Process 0x41 packet (battery, temp, GPIO data)
        a4_data = None
        if 0x41 in packets:
            a4_data = self._process_a4_packet(packets[0x41])
            if a4_data:
                # Extract individual scalar values
                telemetry_data["voltage"] = a4_data.get("voltage")
                telemetry_data["current"] = a4_data.get("current")
                telemetry_data["avg_power"] = a4_data.get("avg_power")
                telemetry_data["remaining"] = a4_data.get("remaining")
                telemetry_data["temperature"] = a4_data.get("temperature")
                telemetry_data["fsr1"] = a4_data.get("fsr1")
                telemetry_data["fsr2"] = a4_data.get("fsr2")
                telemetry_data["button"] = a4_data.get("button")
                telemetry_data["AccelRate"] = a4_data.get("AccelRate")
                telemetry_data["GyroRate"] = a4_data.get("GyroRate")
                telemetry_data["AccelDownsample"] = a4_data.get("AccelDownsample")
                telemetry_data["GyroDownsample"] = a4_data.get("GyroDownsample")
        
        # Process 0x45 packet (pressure data)
        a5_data = None
        if 0x45 in packets:
            a5_data = self._process_a5_packet(packets[0x45])
            if a5_data:
                # Extract pressure array and timestamps
                telemetry_data["Pressure"] = a5_data.get("Pressure", [])
                telemetry_data["pressureTs"] = a5_data.get("pressureTs", [])
        
        # Process 0xA1 packet (accelerometer/gyroscope data) - using already processed data
        if 0xA1 in packets:
            a1_data = self._process_a1_packet(packets[0xA1], a5_data, a4_data)
            if a1_data:
                # Extract accelerometer and gyroscope arrays with enhanced timestamps
                telemetry_data["Ax"] = a1_data.get("Ax", [])
                telemetry_data["Ay"] = a1_data.get("Ay", [])
                telemetry_data["Az"] = a1_data.get("Az", [])
                telemetry_data["Gx"] = a1_data.get("Gx", [])
                telemetry_data["Gy"] = a1_data.get("Gy", [])
                telemetry_data["Gz"] = a1_data.get("Gz", [])
                telemetry_data["AccelTs"] = a1_data.get("AccelTimestamp", [])  # Separate accel timestamps
                telemetry_data["GyroTs"] = a1_data.get("GyroTimestamp", [])   # Separate gyro timestamps
                telemetry_data["AgTs"] = a1_data.get("Timestamp", [])  # Legacy field for backward compatibility
        
        # Send incomplete telemetry data to ThingsBoard
        if self.network_module:
            logger.info(f"Sending incomplete telemetry data for sensor {mac_address} to ThingsBoard")
            self.network_module.send_data_to_thingsboard(telemetry_data)
        else:
            logger.warning("Network module not available, cannot send incomplete telemetry data")
    
    def _check_complete_packet_set(self, mac_address):
        """Check if we have received all expected packet types for a MAC address."""
        with self.lock:
            received_types = set(self.packet_buffers[mac_address].keys())
            return received_types == self.expected_packet_types
    
    def _combine_and_upload_packets(self, mac_address):
        """Combine all packets for a MAC address and upload to ThingsBoard."""
        with self.lock:
            if mac_address not in self.packet_buffers:
                logger.error(f"No packet buffer found for MAC {mac_address}")
                return
            
            packets = self.packet_buffers[mac_address].copy()
            
            # Clear the buffers for this MAC
            del self.packet_buffers[mac_address]
            del self.packet_timestamps[mac_address]
        
        logger.info(f"Combining and uploading complete packet set for MAC {mac_address}")
        
        # Initialize the structured telemetry format with complete data metadata
        telemetry_data = {
            "Location": self.gateway_id,
            "sensor_mac": mac_address,
            "data_complete": True,  # Flag indicating complete data
            "missing_packet_types": [],  # No missing packets
        }
        
        # Process 0x41 packet (battery, temp, GPIO data)
        a4_data = None
        if 0x41 in packets:
            a4_data = self._process_a4_packet(packets[0x41])
            if a4_data:
                # Extract individual scalar values
                telemetry_data["voltage"] = a4_data.get("voltage")
                telemetry_data["current"] = a4_data.get("current")
                telemetry_data["avg_power"] = a4_data.get("avg_power")
                telemetry_data["remaining"] = a4_data.get("remaining")
                telemetry_data["temperature"] = a4_data.get("temperature")
                telemetry_data["fsr1"] = a4_data.get("fsr1")
                telemetry_data["fsr2"] = a4_data.get("fsr2")
                telemetry_data["button"] = a4_data.get("button")
                telemetry_data["AccelRate"] = a4_data.get("AccelRate")
                telemetry_data["GyroRate"] = a4_data.get("GyroRate")
                telemetry_data["AccelDownsample"] = a4_data.get("AccelDownsample")
                telemetry_data["GyroDownsample"] = a4_data.get("GyroDownsample")
        
        # Process 0x45 packet (pressure data)
        a5_data = None
        if 0x45 in packets:
            a5_data = self._process_a5_packet(packets[0x45])
            if a5_data:
                # Extract pressure array and timestamps
                telemetry_data["Pressure"] = a5_data.get("Pressure", [])
                telemetry_data["pressureTs"] = a5_data.get("pressureTs", [])
        
        # Process 0xA1 packet (accelerometer/gyroscope data) - using already processed data
        if 0xA1 in packets:
            a1_data = self._process_a1_packet(packets[0xA1], a5_data, a4_data)
            if a1_data:
                # Extract accelerometer and gyroscope arrays with enhanced timestamps
                telemetry_data["Ax"] = a1_data.get("Ax", [])
                telemetry_data["Ay"] = a1_data.get("Ay", [])
                telemetry_data["Az"] = a1_data.get("Az", [])
                telemetry_data["Gx"] = a1_data.get("Gx", [])
                telemetry_data["Gy"] = a1_data.get("Gy", [])
                telemetry_data["Gz"] = a1_data.get("Gz", [])
                telemetry_data["AccelTs"] = a1_data.get("AccelTimestamp", [])  # Separate accel timestamps
                telemetry_data["GyroTs"] = a1_data.get("GyroTimestamp", [])   # Separate gyro timestamps
                telemetry_data["AgTs"] = a1_data.get("Timestamp", [])  # Legacy field for backward compatibility
        
        # Send structured telemetry data to ThingsBoard
        if self.network_module:
            logger.info(f"Sending structured telemetry data for sensor {mac_address} to ThingsBoard")
            logger.info(f"Structured telemetry data: {json.dumps(telemetry_data, indent=2)}")
            self.network_module.send_data_to_thingsboard(telemetry_data)
        else:
            logger.warning("Network module not available, cannot send telemetry data")
    
    def _process_a1_packet(self, parsed_data, pressure_data=None, gpio_data=None):
        """Process 0xA1 packet data (accelerometer/gyroscope) with enhanced timestamp synthesis."""
        payload = parsed_data.get('payload', {})
        
        # Extract arrays from payload
        timestamp_array = payload.get('Timestamp', [])
        ax_array = payload.get('Ax', [])
        ay_array = payload.get('Ay', [])
        az_array = payload.get('Az', [])
        gx_array = payload.get('Gx', [])
        gy_array = payload.get('Gy', [])
        gz_array = payload.get('Gz', [])
        
        # TEMPORARY FIX: Reorganize IMU data to fix ordering issue
        # Find the common zero point across all IMU arrays
        zero_point_index = self._find_imu_zero_point(ax_array, ay_array, az_array, gx_array, gy_array, gz_array)
        
        if zero_point_index is not None:
            logger.info(f"Applying temporary IMU data reorganization fix at zero point index {zero_point_index}")
            # Apply reorganization to all IMU data arrays
            ax_array = self._reorganize_imu_data(ax_array, zero_point_index)
            ay_array = self._reorganize_imu_data(ay_array, zero_point_index)
            az_array = self._reorganize_imu_data(az_array, zero_point_index)
            gx_array = self._reorganize_imu_data(gx_array, zero_point_index)
            gy_array = self._reorganize_imu_data(gy_array, zero_point_index)
            gz_array = self._reorganize_imu_data(gz_array, zero_point_index)
            logger.info("IMU data reorganization completed")
        else:
            logger.info("No common zero point found, using original IMU data order")
        
        # Determine sample count from the largest data array
        sample_count = max(
            len(ax_array), len(ay_array), len(az_array),
            len(gx_array), len(gy_array), len(gz_array)
        )
        
        if sample_count <= 0:
            logger.warning("No sensor data arrays found in 0xA1 packet")
            return None
        
        # Try pressure-based timestamp synthesis for accelerometer
        accel_timestamps = None
        if self._validate_pressure_timing_data(pressure_data, gpio_data, "accelerometer"):
            try:
                last_pressure_ts = pressure_data['pressureTs'][-1]
                accel_rate = gpio_data['AccelRate']
                accel_downsample = gpio_data['AccelDownsample']
                
                accel_timestamps = self._synthesize_sensor_timestamps_from_pressure(
                    sample_count, last_pressure_ts, accel_rate, accel_downsample, "accelerometer"
                )
                logger.info("Using pressure-based accelerometer timestamps")
            except Exception as e:
                logger.warning(f"Failed to generate pressure-based accelerometer timestamps: {e}")
                accel_timestamps = None
        
        # Fallback to current synthesis for accelerometer
        if not accel_timestamps:
            logger.info("Using fallback timestamp synthesis for accelerometer")
            accel_timestamps = self.synthesize_timestamps(sample_count)
        
        # Try pressure-based timestamp synthesis for gyroscope
        gyro_timestamps = None
        if self._validate_pressure_timing_data(pressure_data, gpio_data, "gyroscope"):
            try:
                last_pressure_ts = pressure_data['pressureTs'][-1]
                gyro_rate = gpio_data['GyroRate']
                gyro_downsample = gpio_data['GyroDownsample']
                
                gyro_timestamps = self._synthesize_sensor_timestamps_from_pressure(
                    sample_count, last_pressure_ts, gyro_rate, gyro_downsample, "gyroscope"
                )
                logger.info("Using pressure-based gyroscope timestamps")
            except Exception as e:
                logger.warning(f"Failed to generate pressure-based gyroscope timestamps: {e}")
                gyro_timestamps = None
        
        # Fallback to current synthesis for gyroscope
        if not gyro_timestamps:
            logger.info("Using fallback timestamp synthesis for gyroscope")
            gyro_timestamps = self.synthesize_timestamps(sample_count)
        
        return {
            "Timestamp": accel_timestamps,  # Legacy field for backward compatibility
            "AccelTimestamp": accel_timestamps,  # Separate accelerometer timestamps
            "GyroTimestamp": gyro_timestamps,   # Separate gyroscope timestamps
            "Ax": ax_array,
            "Ay": ay_array,
            "Az": az_array,
            "Gx": gx_array,
            "Gy": gy_array,
            "Gz": gz_array,
        }
    
    def _map_accel_odr(self, raw_value):
        """Map accelerometer ODR byte value to actual Hz."""
        accel_odr_map = {
            0x01: 0.78, 0x02: 1.56, 0x03: 3.12, 0x04: 6.25,
            0x05: 12.5, 0x06: 25, 0x07: 50, 0x08: 100,
            0x09: 200, 0x0A: 400, 0x0B: 800, 0x0C: 1600
        }
        return accel_odr_map.get(raw_value, raw_value)  # Fallback to raw if unknown
    
    def _map_gyro_odr(self, raw_value):
        """Map gyroscope ODR byte value to actual Hz."""
        gyro_odr_map = {
            0x06: 25, 0x07: 50, 0x08: 100, 0x09: 200,
            0x0A: 400, 0x0B: 800, 0x0C: 1600, 0x0D: 3200
        }
        return gyro_odr_map.get(raw_value, raw_value)  # Fallback to raw if unknown
    
    def _map_downsample_factor(self, raw_value):
        """Map FIFO downsample byte value to actual factor."""
        downsample_map = {
            0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128
        }
        return downsample_map.get(raw_value, raw_value)  # Fallback to raw if unknown
    
    def _find_imu_zero_point(self, ax_array, ay_array, az_array, gx_array, gy_array, gz_array):
        """
        Find the index where ALL IMU data arrays have zero values.
        
        Args:
            ax_array, ay_array, az_array: Accelerometer data arrays
            gx_array, gy_array, gz_array: Gyroscope data arrays
        
        Returns:
            Index where all arrays are zero, or None if not found
        """
        # Get the minimum length to avoid index errors
        arrays = [ax_array, ay_array, az_array, gx_array, gy_array, gz_array]
        if not all(arrays):  # Check if any array is empty
            return None
            
        min_length = min(len(arr) for arr in arrays if arr)
        if min_length == 0:
            return None
        
        # Find index where ALL arrays have zero values
        for i in range(min_length):
            if (ax_array[i] == 0 and ay_array[i] == 0 and az_array[i] == 0 and
                gx_array[i] == 0 and gy_array[i] == 0 and gz_array[i] == 0):
                logger.info(f"Found common zero point at index {i} across all IMU arrays")
                return i
        
        logger.warning("No common zero point found across all IMU arrays")
        return None
    
    def _reorganize_imu_data(self, data_array, zero_point_index):
        """
        Reorganize IMU data array based on the common zero point.
        
        Args:
            data_array: List of sensor values (e.g., Ax, Ay, Az, Gx, Gy, Gz)
            zero_point_index: Index where all IMU arrays are zero
        
        Returns:
            Reorganized data array
        """
        if not data_array or len(data_array) <= 1 or zero_point_index is None:
            return data_array
        
        if zero_point_index >= len(data_array):
            logger.warning(f"Zero point index {zero_point_index} is beyond array length {len(data_array)}")
            return data_array
        
        # Split the data at the zero point
        # first_part: from start to zero point (this should be the end)
        # second_part: from zero point to end (this should be the beginning)
        first_part = data_array[:zero_point_index]  # This becomes the end
        zero_value = [data_array[zero_point_index]]  # The zero point
        second_part = data_array[zero_point_index + 1:]  # This becomes the beginning
        
        # Reorganize: second_part + zero_value + first_part
        reorganized_data = second_part + zero_value + first_part
        
        return reorganized_data
    
    def _process_a4_packet(self, parsed_data):
        """Process 0x41 packet data (battery, temp, GPIO)."""
        payload = parsed_data.get('payload', {})
        logger.info("received for 41: " + json.dumps(parsed_data, indent=2))
        
        # Extract and map the new BMI2 fields
        raw_accel_rate = payload.get("AccelRate")
        raw_gyro_rate = payload.get("GyroRate")
        raw_accel_downsample = payload.get("AccelDownsample")
        raw_gyro_downsample = payload.get("GyroDownsample")
        
        # Apply BMI2 mappings to convert raw bytes to actual values
        mapped_accel_rate = self._map_accel_odr(raw_accel_rate) if raw_accel_rate is not None else None
        mapped_gyro_rate = self._map_gyro_odr(raw_gyro_rate) if raw_gyro_rate is not None else None
        mapped_accel_downsample = self._map_downsample_factor(raw_accel_downsample) if raw_accel_downsample is not None else None
        mapped_gyro_downsample = self._map_downsample_factor(raw_gyro_downsample) if raw_gyro_downsample is not None else None
        
        # Extract individual scalar values from payload
        return {
            "voltage": payload.get("voltage"),
            "current": payload.get("current"),
            "avg_power": payload.get("avg_power"),
            "remaining": payload.get("remaining"),
            "temperature": payload.get("temperature"),
            "fsr1": payload.get("fsr1"),
            "fsr2": payload.get("fsr2"),
            "button": payload.get("button"),
            "AccelRate": mapped_accel_rate,
            "GyroRate": mapped_gyro_rate,
            "AccelDownsample": mapped_accel_downsample,
            "GyroDownsample": mapped_gyro_downsample,
        }
    
    def _process_a5_packet(self, parsed_data):
        """Process 0x45 packet data (pressure sensor data)."""
        logger.info("Processing 0x45 packet (pressure data)")
        payload = parsed_data.get('payload', {})
        logger.info("received for 45: " + json.dumps(parsed_data, indent=2))
        
        # Extract pressure array and timestamps from payload
        # The C code uses "Pressure" (capital P) for the pressure array
        pressure_array = payload.get('Pressure', [])
        pressure_timestamps = payload.get('Timestamp', [])  # 'Timestamp' from decoded data
        
        return {
            "Pressure": pressure_array,  # Use capital P to match final telemetry format
            "pressureTs": pressure_timestamps,  # Map to pressureTs for final upload
        }
    
    def process_sensor_data(self, sensor_mac, parsed_data):
        """
        Buffer and process parsed sensor data. Wait for all expected packet types
        (0xA1, 0x41, 0x45) before combining and uploading to ThingsBoard.
        If timeout occurs, incomplete packet sets are uploaded with metadata indicating missing packets.
        
        Args:
            sensor_mac: MAC address of the sensor
            parsed_data: Dictionary containing parsed data from BLE module with keys:
                        - packet_type: Type of packet (0xA1, 0x41, 0x45)
                        - hardware_id: Hardware identifier of the sensor
                        - payload: Dictionary containing sensor data
        """
        packet_type = parsed_data.get('packet_type')
        
        if packet_type not in self.expected_packet_types:
            logger.warning(f"Unexpected packet type {hex(packet_type) if packet_type else 'None'} for sensor {sensor_mac}")
            logger.info("received data: " + json.dumps(parsed_data, indent=2))
            return
        
        logger.info(f"Buffering packet type {hex(packet_type)} for sensor {sensor_mac}")
        
        # Store the packet in the buffer with thread safety
        with self.lock:
            self.packet_buffers[sensor_mac][packet_type] = parsed_data
            self.packet_timestamps[sensor_mac][packet_type] = time.time()
            
            # Log current buffer status
            received_types = set(self.packet_buffers[sensor_mac].keys())
            missing_types = self.expected_packet_types - received_types
            logger.info(f"Sensor {sensor_mac} buffer status: received {[hex(t) for t in received_types]}, "
                       f"missing {[hex(t) for t in missing_types]}")
        
        # Check if we have all expected packet types for this MAC
        if self._check_complete_packet_set(sensor_mac):
            logger.info(f"Complete packet set received for sensor {sensor_mac}, combining and uploading")
            self._combine_and_upload_packets(sensor_mac)
        else:
            logger.info(f"Waiting for more packets for sensor {sensor_mac}")
    

