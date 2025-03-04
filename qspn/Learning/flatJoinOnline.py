from Learning.qspnJoinReader import multi_table_workload_csv_reader, multi_table_dataset_csv_reader, workload_data_columns_stats
from Learning.flatJoinBase import MultiFLAT

def MultiFLAT_inference(mflat, query):
    #find out all m nodes to calc by query
    #gen sub-q query_i for FSPN of node_i, calc prob_node_i(query_i)
    #calc total_size of outer_join(all calced nodes)
    #result = total_size * prods(prob_node_0(query_0), ..., prob_node_m-1(query_m-1))
    pass