import pandas as pd
import numpy as np

import os
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2x_dataset_Main_run.csv")
df = pd.read_csv(DATA_PATH)

remaining = [
    'timestamp','pos_x','pos_y','speed_mps','distance_moved','position_changes',
    'avg_speed','pkt_size_bytes','my_beacons_sent','my_beacons_received',
    'beacon_send_rate','beacon_recv_rate','total_bytes_sent','total_bytes_rcvd',
    'avg_pkt_size_rcvd','throughput_bps','send_recv_ratio','active_duration',
    'beacon_regularity','global_beacons_sent','global_beacons_rcvd',
    'global_attack_pkts','global_dropped','network_load',
    'norm_send_rate','norm_recv_rate','norm_bytes_sent','norm_speed'
]

bh_df = df[df['label'] == 'blackhole']
other_df = df[df['label'] != 'blackhole']
sybil_df = df[df['label'] == 'sybil']
normal_df = df[df['label'] == 'normal']
ddos_df = df[df['label'] == 'ddos']

print("=== Columns where blackhole is ALL ZEROS (simulation artifact) ===")
for col in remaining:
    if bh_df[col].max() == 0 and bh_df[col].min() == 0:
        others_mean = other_df[col].mean()
        print(f"  {col:25s}  blackhole=[0,0]  others_mean={others_mean:.2f}")

print()
print("=== Columns where sybil is clearly separable (>2x or <0.5x normal AND ddos) ===")
for col in remaining:
    s = sybil_df[col].mean()
    n = normal_df[col].mean()
    d = ddos_df[col].mean()
    if n > 0 and d > 0:
        rn = s / n
        rd = s / d
        if (rn > 2 and rd > 2) or (rn < 0.5 and rd < 0.5):
            print(f"  {col:25s}  sybil={s:.1f}  normal={n:.1f}  ddos={d:.1f}")

print()
print("=== global_* columns (network-wide; unrealistic for single-node IDS) ===")
for col in remaining:
    if col.startswith('global_'):
        g = df.groupby('label')[col].mean()
        print(f"  {col:25s}  {dict(g.items())}")
