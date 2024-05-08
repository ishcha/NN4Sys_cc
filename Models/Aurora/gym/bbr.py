import time

class BBR:
    def __init__(self):
        self.bw = 0  # Estimated bottleneck bandwidth
        self.rtt = float('inf')  # Minimum round trip time
        self.pacing_rate = 1.0
        self.cwnd = 10  # Congestion window in packets
        self.last_sent_time = None

    def update_rtt(self, rtt_sample):
        self.rtt = min(self.rtt, rtt_sample)

    def update_bandwidth(self, delivered_bytes, interval):
        current_bw = delivered_bytes / interval 
        self.bw = max(self.bw, current_bw)
        self.pacing_rate = self.bw

    def on_ack(self, acked_bytes, rtt_sample):
        now = time.time()
        if self.last_sent_time:
            interval = now - self.last_sent_time
            self.update_bandwidth(acked_bytes, interval)
        self.update_rtt(rtt_sample)
        self.last_sent_time = now
        self.cwnd += acked_bytes / self.cwnd

    def send_packet(self):
        # Simulate sending a packet
        time.sleep(1 / self.pacing_rate)
        print(f"Sending packet at {time.time()} with pacing rate {self.pacing_rate}")

# Example usage
bbr = BBR()
bbr.on_ack(1000, 0.1)  # Acknowledge 1000 bytes with a RTT sample of 0.1 seconds
for _ in range(5):
    bbr.send_packet()








import time
import socket
import csv
from threading import Thread

def read_traces(trace_file):
    traces = []
    with open(trace_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            time_seconds, bandwidth_kbps = map(float, row)
            traces.append((time_seconds, bandwidth_kbps))
    return traces

class NetworkSimulator:
    def __init__(self, traces):
        self.traces = traces
        self.current_bandwidth = traces[0][1]  # Start with the first trace point

    def adjust_bandwidth(self):
        """ Adjusts bandwidth every second according to the trace data. """
        start_time = time.time()
        for time_seconds, bandwidth_kbps in self.traces:
            while time.time() - start_time < time_seconds:
                time.sleep(0.1)  # Check every 100 ms
            self.current_bandwidth = bandwidth_kbps

    def start_simulation(self):
        """ Starts the bandwidth simulation in a separate thread. """
        thread = Thread(target=self.adjust_bandwidth)
        thread.start()

    def send_data(self, data, client_socket):
        """ Simulates sending data with the current bandwidth limit. """
        bytes_per_second = (self.current_bandwidth * 1000) / 8  # Convert kbps to Bps
        chunk_size = int(bytes_per_second / 10)  # Send data in 10 parts per second
        start_time = time.time()

        for i in range(0, len(data), chunk_size):
            end_time = time.time()
            if end_time - start_time > 1:
                start_time = end_time  # Reset timer every second
            client_socket.sendall(data[i:i + chunk_size])
            time.sleep(1 / 10)  # Wait to send the next chunk

# Usage example
traces = read_traces('network_traces.csv')
simulator = NetworkSimulator(traces)
simulator.start_simulation()

# Server setup (simple example)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 9999))
server_socket.listen(1)
conn, addr = server_socket.accept()

# Assuming data to send
data_to_send = b'Your data payload here' * 1024  # Simulated data
simulator.send_data(data_to_send, conn)

conn.close()
server_socket.close()
