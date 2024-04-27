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
