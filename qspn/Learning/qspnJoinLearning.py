from Learning.qspnJoinBase import MultiQSPN, get_join_table, multi_table_RDC
from Learning.qspnJoinlearningWrapper import learn_FSPN
from Structure.nodes import Context
from Structure.leaves.parametric.Parametric import Categorical
from Learning.qspnJoinReader import multi_table_workload_csv_reader, multi_table_dataset_csv_reader, workload_data_columns_stats, workload_join_pattern_pairs
from Learning.structureLearning import calculate_RDC

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

def qspn2_calc_RDC(bigtable, tbnameA, tbnameB, cols):
    print(tbnameA, 'JOIN', tbnameB)
    print({ith: i for ith, i in enumerate(list(cols))})
    scope = [i for i in range(len(cols))]
    sample_data = bigtable.values.astype(int)
    parametric_types = [Categorical for i in range(len(bigtable.columns))]
    ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)
    rdc_mat, scope_loc, condition_loc = calculate_RDC(sample_data, ds_context, scope, [], 100000)
    print(rdc_mat)
    print(scope_loc)
    print(condition_loc)
    print()

def learn_multi_QSPN2(
        dataset_root,
        workload,
        cols="rdc",
        rows="grid_naive",
        queries="kmeans",
        threshold=0.3,
        rdc_sample_size=100000,
        rdc_strong_connection_threshold=0.75,
        wkld_attr_threshold=0.01,
        wkld_attr_bound=(0.5, 0.1, 0.3),
        multivariate_leaf=True,
        ohe=False,
        leaves=None,
        leaves_corr=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
        updateQSPN_scope=None,
        updateQSPN_workload_all_n=None,
        qdcorr=None,
        qspn_multihist_max_scope_n=1
):
    #workload: (tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    dc, join_graph = workload_data_columns_stats(workload)
    print(join_graph)
    #exit(-1)
    #print(join_graph)
    #exit(-1)
    #read all .csvs (including header in .csv)
    #print(dc)
    #print(len(workload))
    #print(workload[0])
    #print(workload[1])
    #print(workload[-1])
    data_tables = multi_table_dataset_csv_reader(dataset_root, dc) #table_name: data(DataFrames)
    # for i, ii in data_tables.items():
    #     print(i, ii.shape, ii.dtypes)
    #print(data_tables)
    #exit(-1)
    # for i_k, i_v in data_tables.items():
    #     print(i_k)
    #     print(i_v.columns)
    #     print('min:', i_v.min(axis=0))
    #     print('max:', i_v.max(axis=0))
    # train_time = 0
    join_bigtables_start = perf_counter()
    mqspn = MultiQSPN()
    mqspn.init_bigtable()
    join_pattern = workload_join_pattern_pairs(workload)
    #print(join_pattern)
    #exit(-1)
    print('gen joined bigtables...')
    bigtable_names = []
    bigtable_datas = []
    bigtable_dsfs = []
    for i_k, i_v in join_pattern.items():
        assert len(i_v) == 1
        jc = list(i_v.keys())[0]
        tA, tB, jcA, jcB = i_k[0], i_k[1], jc[0].split('.')[1], jc[1].split('.')[1]
        tAB = mqspn.set_bigtable_join_info(tA, jcA, tB, jcB)
        if multi_table_RDC(data_tables[tA], data_tables[tB], jcA, jcB):
            print('{}: {}.{} outer-join {}.{}'.format(tAB, tA, jcA, tB, jcB))
            #x = data_tables[tA].sample(n=10)
            #print(x)
            #x['__row_number__'] = x.index
            #x.set_index(jcA, inplace=True)
            #print(x)
            #print(x.index)
            #print(x.values.astype(int))
            #exit(-1)
            data_AB, dsfA, dsfB = get_join_table(tA, data_tables[tA], jcA, tB, data_tables[tB], jcB)
            mqspn.set_bigtable_columns(tAB, list(data_AB.columns), data_AB)
            bigtable_names.append(tAB)
            bigtable_datas.append(data_AB)
            bigtable_dsfs.append([dsfA, dsfB])
            qspn2_calc_RDC(data_AB, tA, tB, data_AB.columns) #debug
    mqspn.calc_bigtables_domain(bigtable_names, bigtable_datas)
    #print(mqspn.bigtable_domain)
    join_key_mini = float('inf')
    join_key_maxi = float('-inf')
    for i_k, i_v in mqspn.bigtable_domain.items():
        join_key_mini = min(join_key_mini, i_v[mqspn.bigtable_columns[i_k]['__join_key__']][0])
        join_key_maxi = max(join_key_maxi, i_v[mqspn.bigtable_columns[i_k]['__join_key__']][1])
    for i in mqspn.bigtable_columns:
        mqspn.set_bigtable_column_domain(i, '__join_key__', join_key_mini, join_key_maxi)
    print(mqspn.bigtable_domain)
    exit(-1)
    train_time = perf_counter() - join_bigtables_start
    #print(mqspn.bigtable_columns)
    #print(mqspn.bigtable_domain)
    #print(mqspn.bigtable_cardinality)
    #print(mqspn.bigtable_join_info)
    #exit(-1)
    #workload_join_pattern
    #join tables to big_tables
    #train qspn models on big_tables (with AB and BA downscale_factor individually)

    #class MultiQSPN
    #train_time = 0
    '''
    mqspn_init_start = perf_counter()
    mqspn = MultiQSPN()
    for i, data in data_tables.items():
        mqspn.set_table_columns(i, list(data.columns), data)
    mqspn.calc_tables_domain(list(data_tables.keys()), list(data_tables.values()))
    for belong_i, i in join_graph.items():
        print(belong_i)
        mini = float('inf')
        maxi = float('-inf')
        for j in i:
            jt, jc = j.split('.')
            jsc = mqspn.table_columns[jt][jc]
            jmin, jmax = mqspn.table_domain[jt][jsc]
            mini = min(mini, jmin)
            maxi = max(maxi, jmax)
            #print('\t', j, mqspn.table_domain[jt][jsc])
        for j in i:
            jt, jc = j.split('.')
            mqspn.set_table_column_domain(jt, jc, mini, maxi)
    # for i, domi in mqspn.table_domain.items():
    #     print(i)
    #     for j, jsc in mqspn.table_columns[i].items():
    #         print('\t', j, 'scope =', jsc, domi[jsc])
    # exit(-1)
    train_time += perf_counter() - mqspn_init_start
    '''
    #print(mqspn.table_columns)
    #print(mqspn.table_domain)
    #print(mqspn.table_cardinality)
    #print(train_time, 'sec')
    #exit(-1)
    #gen sub-workload for each table
    bigdata_workload = {i: [] for i in bigtable_names} #table_name: [(join_scope_list, query_ndarray)]
    for i in workload:
        #print('query:', i)
        subq_i_bigtable_names = set()
        if len(i[1]) == 0:
            for j in i[2]:
                jt, jc, jop, jv = j[0], j[1], j[2], j[3]
                for k in bigtable_names:
                    if jt in k:
                        subq_i_bigtable_names.add(k)
        else:
            for j in i[1]:
                lt, lc = j[0].split('.')
                rt, rc = j[1].split('.')
                if lt > rt:
                    lt, lc, rt, rc = rt, rc, lt, lc
                assert (lt, rt) in bigtable_names and lc == mqspn.bigtable_join_info[lt][1] and rc == mqspn.bigtable_join_info[rt][1]
                subq_i_bigtable_names.add((lt, rt))
            for j in i[2]:
                jt, jc, jop, jv = j[0], j[1], j[2], j[3]
                in_bigtable = False
                for k in subq_i_bigtable_names:
                    if jt in k:
                        in_bigtable = True
                        break
                assert in_bigtable
        for j in subq_i_bigtable_names:
            #print('bigtable:', j)
            workload_i_join = set()
            for k in i[1]:
                for l in k:
                    lt, lc = l.split('.')
                    if lt in j:
                        assert lc == mqspn.bigtable_join_info[lt][1]
                        workload_i_join.add(mqspn.bigtable_columns[j]['__join_key__'])
            query_ndarray = np.zeros((len(mqspn.bigtable_columns[j]), 2))
            query_ndarray[:, 0] = float('-inf')
            query_ndarray[:, 1] = float('inf')
            for k in i[2]:
                if k[0] in j:
                    if k[1] == mqspn.bigtable_join_info[k[0]][1]:
                        k_col = '__join_key__'
                    else:
                        k_col = '{}.{}'.format(k[0], k[1])
                    k_scope = mqspn.bigtable_columns[j][k_col]
                    if k[2] == '=':
                        query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
                        query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
                    elif k[2] == '<=':
                        query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
                    elif k[2] == '<':
                        query_ndarray[k_scope, 1] = min(k[3] - 1, query_ndarray[k_scope, 1])
                    elif k[2] == '>=':
                        query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
                    elif k[2] == '>':
                        query_ndarray[k_scope, 0] = max(k[3] + 1, query_ndarray[k_scope, 0])
            #print('workload_i_join:', list(workload_i_join))
            #print(query_ndarray)
            bigdata_workload[j].append((list(workload_i_join), query_ndarray))
    #exit(-1)
        # for j in i[0]:
        #     workload_i_join = set()
        #     for k in i[1]:
        #         for l in k:
        #             lt, lc = l.split('.')
        #             if lt == j:

        #                 workload_i_join.add(mqspn.bigtable_join_info[j])
        #     query_ndarray = np.zeros((len(mqspn.table_columns[j]), 2))
        #     query_ndarray[:, 0] = float('-inf')
        #     query_ndarray[:, 1] = float('inf')
        #     for k in i[2]:
        #         if k[0] == j:
        #             k_scope = mqspn.table_columns[j][k[1]]
        #             if k[2] == '=':
        #                 query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
        #                 query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
        #             elif k[2] == '<=':
        #                 query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
        #             elif k[2] == '<':
        #                 query_ndarray[k_scope, 1] = min(k[3] - 1, query_ndarray[k_scope, 1])
        #             elif k[2] == '>=':
        #                 query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
        #             elif k[2] == '>':
        #                 query_ndarray[k_scope, 0] = max(k[3] + 1, query_ndarray[k_scope, 0])
        #     data_workload[j].append((list(workload_i_join), query_ndarray))
    # for i in data_workload:
    #     print(i)
    #     for j in range(0, 5):
    #         print(data_workload[i][j])
    #     print()
    # exit(-1)
    #train one model on each data-workload (build_fjbuckets=domain[table])
    for i, data, dsfs in zip(bigtable_names, bigtable_datas, bigtable_dsfs):
        print(i)
        joined_scope_i = []
        for jc, jsc in mqspn.bigtable_columns[i].items():
            jt = i
            #j = '.'.join([jt, jc])
            if jc == '__join_key__':
                joined_scope_i.append(jsc)
        if len(joined_scope_i) > 0:
            joined_scope_i = set(joined_scope_i)
        else:
            joined_scope_i = None
        #print(i, joined_scope_i)
        #continue
        workload_i = [j[1] for j in bigdata_workload[i]]
        workload_i = np.array(workload_i)
        workload_i_join = [j[0] for j in bigdata_workload[i]]
        #print(workload_i)
        #print(workload_i_join)
        #print(dsfs)
        #print()
        #continue
        #sample_data = data.values.astype(int)
        sample_data = data.values
        print(sample_data.shape)
        print(workload_i.shape, len(workload_i_join), workload_i_join[0 : 10])
        print('joined_tables_name =',list(i))
        #print('joined_downscale_factor_cols =', dsfs)
        #continue
        parametric_types = [Categorical for i in range(len(data.columns))]
        ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)
        qspn = None
        train_i_time = 0
        print(sample_data)
        #input('Press ENTER to continue...')
        qspn, train_i_time = learn_FSPN(
            sample_data,
            ds_context,
            workload=workload_i,
            queries=queries,
            rdc_sample_size=rdc_sample_size,
            rdc_strong_connection_threshold=rdc_strong_connection_threshold,
            multivariate_leaf=multivariate_leaf,
            threshold=threshold,
            wkld_attr_threshold=wkld_attr_threshold,
            wkld_attr_bound=wkld_attr_bound,
            qspn_multihist_max_scope_n=qspn_multihist_max_scope_n,
            build_fjbuckets=mqspn.bigtable_domain[i],
            workload_join=workload_i_join,
            joined_scope=joined_scope_i,
            joined_tables_name=list(i),
            joined_downscale_factor_cols=dsfs
        )
        mqspn.set_bigtable_qspn_model(i, qspn)
        #gen MultiQSPN.calc_table_RDC for each data
        # rdc_start = perf_counter()
        # mqspn.calc_table_RDC(i, sample_data, ds_context, rdc_sample_size)
        # train_i_time += perf_counter() - rdc_start
        train_time += train_i_time
        #input('Press ENTER to continue...')
        print()
    #exit(-1)
    # print(mqspn.table_qspn_model)
    # for i, im in mqspn.table_rdc_adjacency_matrix.items():
    #     print(i)
    #     print(im)
    #exit(-1)
    #return MultiQSPN
    return mqspn, train_time

def learn_multi_QSPN(
        dataset_root,
        workload,
        cols="rdc",
        rows="grid_naive",
        queries="kmeans",
        threshold=0.3,
        rdc_sample_size=100000,
        rdc_strong_connection_threshold=0.75,
        wkld_attr_threshold=0.01,
        wkld_attr_bound=(0.5, 0.1, 0.3),
        multivariate_leaf=True,
        ohe=False,
        leaves=None,
        leaves_corr=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
        updateQSPN_scope=None,
        updateQSPN_workload_all_n=None,
        qdcorr=None,
        qspn_multihist_max_scope_n=1
):
    #workload: (tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    dc, join_graph = workload_data_columns_stats(workload)
    #print(join_graph)
    #exit(-1)
    #read all .csvs (including header in .csv)
    #print(dc)
    #print(len(workload))
    #print(workload[0])
    #print(workload[1])
    #print(workload[-1])
    data_tables = multi_table_dataset_csv_reader(dataset_root, dc) #table_name: data(DataFrames)
    # for i, ii in data_tables.items():
    #     print(i, ii.shape, ii.dtypes)
    #print(data_tables)
    #exit(-1)
    #class MultiQSPN
    train_time = 0
    mqspn_init_start = perf_counter()
    mqspn = MultiQSPN()
    for i, data in data_tables.items():
        mqspn.set_table_columns(i, list(data.columns), data)
    mqspn.calc_tables_domain(list(data_tables.keys()), list(data_tables.values()))
    for belong_i, i in join_graph.items():
        print(belong_i)
        mini = float('inf')
        maxi = float('-inf')
        for j in i:
            jt, jc = j.split('.')
            jsc = mqspn.table_columns[jt][jc]
            jmin, jmax = mqspn.table_domain[jt][jsc]
            mini = min(mini, jmin)
            maxi = max(maxi, jmax)
            #print('\t', j, mqspn.table_domain[jt][jsc])
        for j in i:
            jt, jc = j.split('.')
            mqspn.set_table_column_domain(jt, jc, mini, maxi)
    # for i, domi in mqspn.table_domain.items():
    #     print(i)
    #     for j, jsc in mqspn.table_columns[i].items():
    #         print('\t', j, 'scope =', jsc, domi[jsc])
    # exit(-1)
    train_time += perf_counter() - mqspn_init_start
    #print(mqspn.table_columns)
    #print(mqspn.table_domain)
    #print(mqspn.table_cardinality)
    #print(train_time, 'sec')
    #exit(-1)
    #gen sub-workload for each table
    data_workload = {i: [] for i in data_tables} #table_name: [(join_scope_list, query_ndarray)]
    for i in workload:
        for j in i[0]:
            workload_i_join = set()
            for k in i[1]:
                for l in k:
                    lt, lc = l.split('.')
                    if lt == j:
                        workload_i_join.add(mqspn.table_columns[j][lc])
            query_ndarray = np.zeros((len(mqspn.table_columns[j]), 2))
            query_ndarray[:, 0] = float('-inf')
            query_ndarray[:, 1] = float('inf')
            for k in i[2]:
                if k[0] == j:
                    k_scope = mqspn.table_columns[j][k[1]]
                    if k[2] == '=':
                        query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
                        query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
                    elif k[2] == '<=':
                        query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
                    elif k[2] == '<':
                        query_ndarray[k_scope, 1] = min(k[3] - 1, query_ndarray[k_scope, 1])
                    elif k[2] == '>=':
                        query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
                    elif k[2] == '>':
                        query_ndarray[k_scope, 0] = max(k[3] + 1, query_ndarray[k_scope, 0])
            data_workload[j].append((list(workload_i_join), query_ndarray))
    # for i in data_workload:
    #     print(i)
    #     for j in range(0, 5):
    #         print(data_workload[i][j])
    #     print()
    # exit(-1)
    #train one model on each data-workload (build_fjbuckets=domain[table])
    for i, data in data_tables.items():
        print(i)
        joined_scope_i = []
        for jc, jsc in mqspn.table_columns[i].items():
            jt = i
            j = '.'.join([jt, jc])
            for k in join_graph.values():
                if j in k:
                    joined_scope_i.append(jsc)
                    break
        if len(joined_scope_i) > 0:
            joined_scope_i = set(joined_scope_i)
        else:
            joined_scope_i = None
        #print(i, joined_scope_i)
        #continue
        workload_i = [j[1] for j in data_workload[i]]
        workload_i = np.array(workload_i)
        workload_i_join = [j[0] for j in data_workload[i]]
        sample_data = data.values.astype(int)
        print(sample_data.shape)
        print(workload_i.shape, len(workload_i_join), workload_i_join[0 : 10])
        parametric_types = [Categorical for i in range(len(data.columns))]
        ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)
        qspn = None
        train_i_time = 0
        qspn, train_i_time = learn_FSPN(
            sample_data,
            ds_context,
            workload=workload_i,
            queries=queries,
            rdc_sample_size=rdc_sample_size,
            rdc_strong_connection_threshold=rdc_strong_connection_threshold,
            multivariate_leaf=multivariate_leaf,
            threshold=threshold,
            wkld_attr_threshold=wkld_attr_threshold,
            wkld_attr_bound=wkld_attr_bound,
            qspn_multihist_max_scope_n=qspn_multihist_max_scope_n,
            build_fjbuckets=mqspn.table_domain[i],
            workload_join=workload_i_join,
            joined_scope=joined_scope_i
        )
        mqspn.set_table_qspn_model(i, qspn)
        #gen MultiQSPN.calc_table_RDC for each data
        rdc_start = perf_counter()
        mqspn.calc_table_RDC(i, sample_data, ds_context, rdc_sample_size)
        train_i_time += perf_counter() - rdc_start
        train_time += train_i_time
        print()
    #exit(-1)
    # print(mqspn.table_qspn_model)
    # for i, im in mqspn.table_rdc_adjacency_matrix.items():
    #     print(i)
    #     print(im)
    #exit(-1)
    #return MultiQSPN
    return mqspn, train_time
