import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sensor_data(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create time array (assuming timestamps are in milliseconds)
    time = np.array(data['Timestamp'])
    time = (time - time[0]) / 1000  # Convert to seconds from start
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('Sensor Data Visualization', fontsize=16)
    
    # Plot accelerometer data
    ax1.plot(time, data['Ax'], label='Ax', color='red')
    ax1.plot(time, data['Ay'], label='Ay', color='green')
    ax1.plot(time, data['Az'], label='Az', color='blue')
    ax1.set_title('Accelerometer Data')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Acceleration (g)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-6, 6)
    
    # Plot gyroscope data
    ax2.plot(time, data['Gx'], label='Gx', color='red')
    ax2.plot(time, data['Gy'], label='Gy', color='green')
    ax2.plot(time, data['Gz'], label='Gz', color='blue')
    ax2.set_title('Gyroscope Data')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(-1000, 1000)
    
    # Plot pressure data
    ax3.plot(time, data['Pressure'], label='Pressure', color='purple')
    ax3.set_title('Barometric Pressure')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Pressure (hPa)')
    ax3.grid(True)
    ax3.legend()

    #Plot FSR readings data
    ax4.plot(time, data['FSR'], label='FSR Voltage Value', color='orange')
    ax4.set_title('Force Sensitive Resistor (FSR) Readings')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('FSR Voltage Value (V)')
    ax4.grid(True)
    ax4.legend()
    ax4.set_ylim(0, 1000)  # Assuming FSR voltage is between 0 and 5V
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file)
    plt.close(fig)

def plot_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"plot_{name}.png")
            plot_sensor_data(input_path, output_path)
            print(f"Saved plot for {input_path} to {output_path}")

if __name__ == "__main__":
    input_folder = "processed_final_data"  # Folder with reformatted JSON files
    output_folder = "plots_final_data"  # Folder to save plots
    plot_folder(input_folder, output_folder) 