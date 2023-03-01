import pandas as pd
import json
from pandas import json_normalize

json_string = '{"columns": ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp", "flow_duration", "flow_byts_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "fwd_header_len", "bwd_header_len", "fwd_seg_size_min", "fwd_act_data_pkts", "flow_iat_mean", "flow_iat_max", "flow_iat_min", "flow_iat_std", "fwd_iat_tot", "fwd_iat_max", "fwd_iat_min", "fwd_iat_mean", "fwd_iat_std", "bwd_iat_tot", "bwd_iat_max", "bwd_iat_min", "bwd_iat_mean", "bwd_iat_std", "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "init_fwd_win_byts", "init_bwd_win_byts", "active_max", "active_min", "active_mean", "active_std", "idle_max", "idle_min", "idle_mean", "idle_std", "fwd_byts_b_avg", "fwd_pkts_b_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg", "fwd_blk_rate_avg", "bwd_blk_rate_avg", "fwd_seg_size_avg", "bwd_seg_size_avg", "cwe_flag_count", "subflow_fwd_pkts", "subflow_bwd_pkts", "subflow_fwd_byts", "subflow_bwd_byts"], "data": [["10.0.0.4", "168.63.129.16", 48402, 80, 6, "2023-03-01 15:07:37", 2460.2413177490234, 1131596.311270472, 3251.71353813354, 2032.3209613334625, 1219.3925768000774, 5, 3, 537, 2247, 265.0, 66.0, 107.4, 78.86089018011398, 2107.0, 66.0, 749.0, 960.2565629386069, 2107, 66, 348.0, 667.9462927511463, 446152.25, 100, 60, 20, 1, 351.4630453927176, 1293.6592102050781, 22.88818359375, 448.00605499884875, 2315.044403076172, 1316.5473937988281, 26.226043701171875, 578.761100769043, 505.0464295975474, 1733.0646514892578, 1355.886459350586, 377.1781921386719, 866.5323257446289, 489.35413360595703, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.6, 348.0, 64240, 49153, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 107.4, 749.0, 0, 5, 3, 537, 2247]]}'
dict = json.loads(json_string)
print(dict['columns'])
print(dict['data'][0])
exit()
df1= json_normalize(dict['columns'])
print(df1)

df2= json_normalize(dict['data'])
print(df2)
