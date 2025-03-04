from Learning.qspnJoinReader import multi_table_workload_csv_reader, multi_table_dataset_csv_reader, workload_data_columns_stats

def RDC_two_tables(TA, TB):
    pass

def gen_scattering_coefficient_columns(TA, TB):
    #if A join B, S_A,B_i means count of B_records can join with A_record_i, if A_record_i_join_attr is NaN, S_A,B_i=0
    #S_B,A_i respectively
    
    pass

def outerjoin_two_tables(TA, TB):
    pass

def gen_join_tree(dc, join_graph):
    pass

def learn_multi_FLAT():
    #read dataset tables (csv) and workload trainset
    #gen join tree and join high-corr tables
    #train the FSPN of the big table of each node
    pass