import numpy as np
import pandas as pd
import subprocess
import time
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Headless backend

# Class for Agentic AI Monitor using only current data
class AgenticAIMonitor:
    def __init__(self, seq_len=10):
        """Initialize with LSTM model, scaler, and data buffer for real-time monitoring."""
        self.scaler = MinMaxScaler()
        self.seq_len = seq_len
        self.model = self.build_model()
        self.data_buffer = pd.DataFrame(columns=['PTP_Offset', 'NTP_Latency', 'SNR', 'SyncE_Drift', 'Jitter'])
        self.history = []

    def build_model(self):
        """Build LSTM model to predict PTP offset."""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.seq_len, 5)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fetch_real_data(self):
        """Fetch real-time data from Falcon switch, NTP servers, GNSS antenna, and Benetel."""
        try:
            # PTP offset from Falcon switch (via ptp4l)
            ptp_output = subprocess.run(['ptp4l', '-m'], capture_output=True, text=True, timeout=2).stdout
            ptp_offset = float(ptp_output.split('offset')[-1].split()[0]) if 'offset' in ptp_output else 77.15  # ns

            # NTP latency from ntpstat (fallback to ntpq if not available)
            try:
                ntp_output = subprocess.run(['ntpstat'], capture_output=True, text=True, timeout=2).stdout
                ntp_latency = float(ntp_output.split('time offset')[-1].split()[0]) if 'time offset' in ntp_output else 5.0
            except FileNotFoundError:
                ntpq_output = subprocess.run(['ntpq', '-p'], capture_output=True, text=True, timeout=2).stdout
                ntp_latency = float(ntpq_output.split('\n')[2].split()[7]) if len(ntpq_output.split('\n')) > 2 else 5.0  # ms

            # SNR from GNSS antenna (via gpsmon)
            gnss_output = subprocess.run(['gpsmon', '-n'], capture_output=True, text=True, timeout=2).stdout
            snr = float(gnss_output.split('SNR')[-1].split()[0]) if 'SNR' in gnss_output else 45.0  # dB-Hz

            # SyncE drift from Falcon switch (via ethtool)
            sync_e_output = subprocess.run(['ethtool', '-T', 'eth0'], capture_output=True, text=True, timeout=2).stdout
            sync_e_drift = float(sync_e_output.split('drift')[-1].split()[0]) if 'drift' in sync_e_output else 0.01  # ns/min

            # Jitter from Benetel (via assumed benetel-status)
            benetel_output = subprocess.run(['benetel-status'], capture_output=True, text=True, timeout=2).stdout
            jitter = float(benetel_output.split('jitter')[-1].split()[0]) if 'jitter' in benetel_output else 0.05  # ns

            return pd.DataFrame({
                'PTP_Offset': [ptp_offset],
                'NTP_Latency': [ntp_latency],
                'SNR': [snr],
                'SyncE_Drift': [sync_e_drift],
                'Jitter': [jitter]
            })
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame({
                'PTP_Offset': [77.15],
                'NTP_Latency': [5.0],
                'SNR': [45.0],
                'SyncE_Drift': [0.01],
                'Jitter': [0.05]
            })  # Fallback to reasonable defaults

    def prepare_data(self, data):
        """Prepare data sequences for LSTM."""
        if len(data) < self.seq_len:
            return np.array([]), np.array([])
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.seq_len, len(scaled_data)):
            X.append(scaled_data[i-self.seq_len:i])
            y.append(scaled_data[i, 0])  # Predict PTP offset
        return np.array(X), np.array(y)

    def train(self, X, y, epochs=10, batch_size=32):
        """Train the model with current data."""
        if len(X) == 0 or len(y) == 0:
            print("Insufficient data for training.")
            return None
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.history.append(history.history['loss'])
        return history

    def predict_and_detect(self, X, y, data):
        """Predict PTP offset, detect anomalies, and classify types."""
        if len(X) == 0 or len(y) == 0:
            return np.array([]), [], {}
        predictions = self.model.predict(X)
        errors = np.abs(predictions[:, 0] - y)
        anomalies = np.where(errors > 0.1)[0]  # Threshold for anomaly

        # Classify anomaly types based on current data thresholds
        anomaly_classes = {}
        for idx in anomalies:
            types = []
            global_idx = idx + self.seq_len
            if global_idx >= len(data): continue
            row = data.iloc[global_idx]
            if row['NTP_Latency'] > 10: types.append('NTP_spike')
            if row['PTP_Offset'] > 100: types.append('PTP_drift')
            if row['SNR'] < 42: types.append('GNSS_drop')
            if row['SyncE_Drift'] > 0.1: types.append('SyncE_drift')
            if row['Jitter'] > 0.1: types.append('Jitter_high')
            anomaly_classes[idx] = types if types else ['Normal']

        print(f"Anomalies detected at indices: {anomalies}")
        print(f"Classified anomaly types: {anomaly_classes}")
        return predictions, anomalies, anomaly_classes

    def plot_results(self, predictions, y, anomalies, time_steps):
        """Save plots for training loss and anomaly detection."""
        if self.history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history[-1], label='Training Loss')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_loss.png')
            plt.close()
            print("Training loss plot saved as training_loss.png")

        if len(predictions) > 0 and len(y) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(range(self.seq_len, time_steps), y, label='Actual PTP Offset (Scaled)')
            plt.plot(range(self.seq_len, time_steps), predictions, label='Predicted PTP Offset (Scaled)')
            plt.axvline(x=200, color='r', linestyle='--', label='Potential Anomaly Start')
            for idx in anomalies:
                plt.axvline(x=idx + self.seq_len, color='#FFA500', alpha=0.5, linestyle=':')
            plt.title('Actual vs Predicted PTP Offset with Anomalies')
            plt.xlabel('Time Step')
            plt.ylabel('Scaled PTP Offset')
            plt.legend()
            plt.grid(True)
            plt.savefig('anomaly_detection.png')
            plt.close()
            print("Anomaly detection plot saved as anomaly_detection.png")

    def continuous_monitoring(self, update_interval=300):
        """Continuously monitor and update model with real-time data, generating final diagram."""
        print(f"Starting continuous monitoring with update interval of {update_interval} seconds...")
        while True:
            try:
                # Fetch new data
                new_data = self.fetch_real_data()
                if not new_data.empty and new_data.notna().all().all():  # Ensure valid data
                    self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)

                # Prepare and train if enough data
                if len(self.data_buffer) >= self.seq_len:
                    X, y = self.prepare_data(self.data_buffer)
                    if len(X) > 0 and len(y) > 0:
                        self.train(X, y)
                        predictions, anomalies, classes = self.predict_and_detect(X, y, self.data_buffer)
                        self.plot_results(predictions, y, anomalies, len(self.data_buffer))

                # Check for user stop
                if input("Continue monitoring? (y/n): ").lower() != 'y':
                    print("Monitoring stopped by user. Generating final diagram...")
                    self.generate_final_diagram()
                    break

                time.sleep(update_interval)  # Wait before next update
            except KeyboardInterrupt:
                print("Monitoring stopped by user. Generating final diagram...")
                self.generate_final_diagram()
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(update_interval)

    def generate_final_diagram(self):
        """Generate a final diagram summarizing monitored data."""
        if self.data_buffer.empty:
            print("No data available for final diagram.")
            return

        plt.figure(figsize=(14, 8))
        plt.plot(self.data_buffer.index, self.data_buffer['PTP_Offset'], label='PTP Offset (ns)', color='blue')
        plt.plot(self.data_buffer.index, self.data_buffer['NTP_Latency'], label='NTP Latency (ms)', color='green')
        plt.plot(self.data_buffer.index, self.data_buffer['SNR'], label='SNR (dB-Hz)', color='red')
        plt.plot(self.data_buffer.index, self.data_buffer['SyncE_Drift'], label='SyncE Drift (ns/min)', color='purple')
        plt.plot(self.data_buffer.index, self.data_buffer['Jitter'], label='Jitter (ns)', color='orange')
        plt.title('Summary of Monitored Time Synchronization Metrics')
        plt.xlabel('Time Step')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('final_monitoring_summary.png')
        plt.close()
        print("Final monitoring summary diagram saved as final_monitoring_summary.png")

if __name__ == "__main__":
    monitor = AgenticAIMonitor()
    monitor.continuous_monitoring(update_interval=300)  # 5-minute updates
