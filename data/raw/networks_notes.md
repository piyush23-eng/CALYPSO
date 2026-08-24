# GATE Computer Science - Computer Networks Reference Notes

## Transport Layer and TCP Congestion Control

### TCP Congestion Control Algorithms and Phases
TCP uses window-based congestion control where sender window size is $W = \min(cwnd, rwnd)$, where $cwnd$ is Congestion Window and $rwnd$ is Receiver Window.

1. **Slow Start**:
   - Initial $cwnd = 1 \text{ MSS}$ (or initial value).
   - For every ACK received, $cwnd$ increases by 1 MSS (exponential growth: doubles every RTT).
   - Continues until $cwnd \ge ssthresh$ (Slow Start Threshold) or packet loss occurs.

2. **Congestion Avoidance**:
   - Entered when $cwnd \ge ssthresh$.
   - $cwnd$ increases by $1 \text{ MSS}$ per RTT (linear growth: Additive Increase).
   - For every ACK, $cwnd = cwnd + 1/cwnd$.

3. **Loss Detection & Fast Recovery**:
   - **Timeout (Severe Loss)**:
     - $ssthresh = \max(cwnd / 2, 2 \text{ MSS})$.
     - $cwnd = 1 \text{ MSS}$.
     - Re-enters **Slow Start**.
   - **3 Duplicate ACKs (Mild Loss / Fast Retransmit)**:
     - $ssthresh = \max(cwnd / 2, 2 \text{ MSS})$.
     - $cwnd = ssthresh + 3 \text{ MSS}$ (Fast Recovery: Multiplicative Decrease).
     - Directly enters Congestion Avoidance without resetting to 1 MSS.

## Network Layer and IPv4 CIDR Subnetting

### Classless Inter-Domain Routing (CIDR)
- IP address is 32-bit: `Network ID / Subnet Mask (/n)` + `Host ID (32 - n bits)`.
- Total available IP addresses in `/n` subnet = $2^{32 - n}$.
- Total usable host IP addresses = $2^{32 - n} - 2$ (excluding Network ID and Directed Broadcast Address).

## Data Link Layer and Flow Control

### Sliding Window Protocols
1. **Stop-and-Wait**:
   - Sender Window $W_s = 1$, Receiver Window $W_r = 1$.
   - Efficiency $\eta = \frac{1}{1 + 2a}$, where $a = \frac{T_p}{T_t} = \frac{\text{Propagation Delay}}{\text{Transmission Delay}}$.
2. **Go-Back-N (GBN)**:
   - $W_s = N$, $W_r = 1$.
   - Sequence number space required $\ge N + 1$ ($k$ bits $\implies N \le 2^k - 1$).
   - Uses cumulative ACKs. If packet lost, entire window retransmitted.
   - Efficiency $\eta = \frac{N}{1 + 2a}$.
3. **Selective Repeat (SR)**:
   - $W_s = N$, $W_r = N$.
   - Sequence number space required $\ge 2N$ ($k$ bits $\implies N \le 2^{k-1}$).
   - Uses individual ACKs. Only lost packets retransmitted.
