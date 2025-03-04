import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def get_workload_attr_matrix(
    local_workload,
    scope,
    local_workload_join=None,
    num_queries=None
):
    wkld_attr_adjacency_matrix = np.zeros([len(scope), len(scope)])
    wkld_in_scope = local_workload[:, scope, :]

    mask = (wkld_in_scope[:,:,0] != -np.inf) | (wkld_in_scope[:,:,1] != np.inf)
    mask = mask.astype(int)

    for i in range(mask.shape[0]):
        query_attr = np.where(mask[i]==1)[0]
        if len(query_attr) == 0:
            continue
        if len(query_attr) == 1:
            wkld_attr_adjacency_matrix[query_attr[0]][query_attr[0]] += 1
        for j in range(len(query_attr)):
            for k in range(j, len(query_attr)):
                aj = query_attr[j]
                ak = query_attr[k]
                wkld_attr_adjacency_matrix[aj][ak] += 1
                wkld_attr_adjacency_matrix[ak][aj] += 1
    #join predicates
    if local_workload_join is not None:
        assert len(local_workload_join) == len(local_workload)
        assert num_queries is not None
        for i in local_workload_join:
            for j in i:
                for k in i:
                    if j != k:
                        wkld_attr_adjacency_matrix[j][k] = num_queries
    
    return wkld_attr_adjacency_matrix

def get_workload_by_scope(scope_slice, workload, workload_join=None):
    wkidx = np.zeros(workload.shape[0]).astype(bool)
    for s in scope_slice:
        wkidx = wkidx | (workload[:, s, 0] != -np.inf) | (workload[:, s, 1] != np.inf)
    if workload_join is None:
        local_workload = workload[wkidx]
        return local_workload
    else:
        #join predicates
        assert len(workload_join) == len(workload)
        for i, q in enumerate(workload_join):
            for j in q:
                if j in scope_slice:
                    wkidx[i] = True
                    break
        local_workload = workload[wkidx]
        local_workload_join = []
        for ith, i in enumerate(wkidx):
            if i:
                local_workload_join.append(workload_join[ith])
        assert len(local_workload) == len(local_workload_join)
        return local_workload, local_workload_join

def get_workload_by_data(data_slice, scope_slice, workload, workload_join=None):
    wkidx = np.zeros(workload.shape[0]).astype(bool)
    data_max = np.max(data_slice, axis=0)
    data_min = np.min(data_slice, axis=0)
    for wid in range(workload.shape[0]):
        query = workload[wid]
        flag = True
        for i, s in enumerate(scope_slice):
            if query[s][0] > data_max[i] or query[s][1] < data_min[i]:
                flag = False
                break
        if flag:
            wkidx[wid] = True 
    if workload_join is None:
        local_workload = workload[wkidx]
        return local_workload
    else:
        local_workload = workload[wkidx]
        local_workload_join = []
        for ith, i in enumerate(wkidx):
            if i:
                local_workload_join.append(workload_join[ith])
        assert len(local_workload) == len(local_workload_join)
        return local_workload, local_workload_join

def get_workload_by_datadom(data_min, data_max, scope_slice, workload):
    wkidx = np.zeros(workload.shape[0]).astype(bool)
    for wid in range(workload.shape[0]):
        query = workload[wid]
        flag = True
        for i, s in enumerate(scope_slice):
            if query[s][0] > data_max[i] or query[s][1] < data_min[i]:
                flag = False
                break
        if flag:
            wkidx[wid] = True 
    local_workload = workload[wkidx]
    return local_workload

def split_queries_by_clusters(workload, clusters, scope, centers=None):
    unique_clusters = np.sort(np.unique(clusters))
    #print(clusters)
    #print(len(clusters))
    #print(len(workload))
    #print(workload[0])
    #print(workload[1])
    #print(workload[2])
    #exit(-1)
    assert centers is None or len(unique_clusters) == centers.shape[0], \
        f"workload shape is {workload.shape}, unique clusters are {unique_clusters} and centers shape is {centers.shape}"
    result = []
    #print(centers)
    for i, uc in enumerate(unique_clusters):
        local_workload = workload[clusters == uc, :]
        proportion = local_workload.shape[0] / workload.shape[0]
        if centers is not None:
            result.append((local_workload, scope, proportion, centers[i]))
        else:
            result.append((local_workload, scope, proportion))
        print(len(local_workload), scope, proportion, centers[i])
    #print(result[0])
    #exit(-1)
    return result

def bitset_intersectbits(a, b):
    assert len(a) == len(b)
    cnt = 0
    for i in range(len(a)):
        if a[i] == b[i] == '1':
            cnt += 1
    return cnt

def qsplit_inference_cluster_select(scope, cluster, query):
    return 0.0

def qsplit_qspnupdate_add_cluster_center_encoder(scope, center):
    encod = np.array(list(map(int, list(center))))
    return encod

def qsplit_train_cluster_encoder(scope, r_clusters):
    encod = [np.array(list(map(int, list(j)))) for j in r_clusters]
    return encod

def qsplit_train_cluster_decoder(scope, r_clusters):
    a2l = [list(map(list, i)) for i in r_clusters]
    i2s = [[''.join(list(map(str, j))) for j in i] for i in a2l]
    decod = i2s
    return decod

def split_queries_by_maxcut_point_encoder(point):
    return ''.join(point)

def split_queries_by_maxcut_point_decoder(point):
    return list(point)

def split_queries_by_maxcut_clusters(workload, clusters, scope, centers, workload_join=None):
    #(local_workload, scope, proportion, centers[i])
    result = []
    r_local_workload = [[] for i in range(len(centers))]
    #join predicates
    if workload_join is not None:
        r_local_workload_join = [[] for i in range(len(centers))]
    for i, q in enumerate(workload):
        r_local_workload[clusters[i]].append(q)
        #join predicates
        if workload_join is not None:
            r_local_workload_join[clusters[i]].append(workload_join[i])
    for i, c in enumerate(centers):
        r_scope = scope
        r_proportion = len(r_local_workload[i]) / len(workload)
        if workload_join is None:
            result.append((np.array(r_local_workload[i]), r_scope, r_proportion, centers[i]))
        else:
            #join predicates
            result.append((np.array(r_local_workload[i]), r_scope, r_proportion, centers[i], r_local_workload_join[i]))
    return result

MAXCUT_K = 7
def get_split_queries_MaxCut_new(workload, scope, workload_join=None):
    ### step1: cluster2points
    clusters = {}
    for i, q in enumerate(workload):
        pointi = ['1'] * len(scope)
        for j, c in enumerate(scope):
            if q[c][0] == float('-inf') and q[c][1] == float('inf'):
                pointi[j] = '0'
        #join predicates
        if workload_join is not None:
            for j in workload_join[i]:
                pointi[j] = '1'
        spointi = split_queries_by_maxcut_point_encoder(pointi)
        if not clusters.__contains__(spointi):
            clusters[spointi] = [i]
        else:
            clusters[spointi].append(i)
    #for i in clusters:
    #    print(i, len(clusters[i]))
    #exit(-1)
    ### step2: build graph
    V = []
    V_weight = []
    E = []
    Esum = 0
    for i in clusters:
        V.append(i)
    GA = np.zeros((len(V), len(V)))
    for i in range(len(V)):
        for j in range(len(V)):
            if i < j:
                GA[i][j] = bitset_intersectbits(V[i], V[j]) * (len(clusters[V[i]]) +  len(clusters[V[j]]))
                GA[j][i] = GA[i][j]
                E.append((GA[i][j], i, j))
                Esum += GA[i][j]
    for i in range(len(V)):
        V_weight.append([i, 0])
        for j in range(len(V)):
            V_weight[i][1] += GA[i][j]
    V_weight = sorted(V_weight, reverse=True, key=lambda vw : vw[1])
    E = sorted(E, reverse=True, key=lambda ew : ew[0])
    #print(V)
    #print(V_weight)
    #print(E)
    #print(GA)
    ### step3: cut max-K E
    cutset = []
    vis_V = set()
    maxKE = set()
    for i in range(min(MAXCUT_K, len(E))):
        if E[i][0] < 1:
            break
        maxKE.add((E[i][1], E[i][2]))
    #print(maxKE)
    #exit(-1)
    ### step4: cut the graph left
    E_sum_opt = 0
    for i in V_weight:
        if i[0] not in vis_V:
            opt_j = None
            opt = None
            for j in range(len(cutset)):
                cost = 0
                cutted = False
                for k in cutset[j]:
                    if (i[0], k) in maxKE or (k, i[0]) in maxKE:
                        cutted = True
                        break
                    cost += GA[i[0]][k]
                if cutted:
                    continue
                if opt is None or cost < opt:
                    opt_j = j
                    opt = cost
            if opt_j is None:
                cutset.append([i[0]])
            else:
                cutset[opt_j].append(i[0])
                E_sum_opt += opt
            vis_V.add(i[0])
    #print(cutset)
    ### step5: final
    #r_score, r_clusters, r_centers
    r_clusters = [None] * len(workload)
    r_centers = []
    maxcut_point = []
    for i, c in enumerate(cutset):
        maxcut_point.append({})
        r_centers.append(qsplit_train_cluster_encoder(scope, [V[j] for j in c]))
        for j in c:
            maxcut_point[-1][V[j]] = len(clusters[V[j]])
            for k in clusters[V[j]]:
                r_clusters[k] = i
    r_score = (Esum - E_sum_opt) / max(1, Esum)
    #print(clusters)
   # maxcut_point = {i: len(clusters[i]) for i in clusters}
    print(r_score, Esum, E_sum_opt, len(cutset))
    #print(r_centers)
    #exit(-1)
    
    return r_score, r_clusters, r_centers, maxcut_point

def get_split_queries_MaxCut_old(workload, scope):
    ### step1: cluster2points
    clusters = {}
    for i, q in enumerate(workload):
        pointi = ['1'] * len(scope)
        for j, c in enumerate(scope):
            if q[c][0] == float('-inf') and q[c][1] == float('inf'):
                pointi[j] = '0'
        spointi = split_queries_by_maxcut_point_encoder(pointi)
        if not clusters.__contains__(spointi):
            clusters[spointi] = [i]
        else:
            clusters[spointi].append(i)
    #for i in clusters:
    #    print(i, len(clusters[i]))
    #exit(-1)
    ### step2: build graph
    V = []
    V_weight = []
    E = []
    Esum = 0
    for i in clusters:
        V.append(i)
    GA = np.zeros((len(V), len(V)))
    for i in range(len(V)):
        for j in range(len(V)):
            if i < j:
                GA[i][j] = bitset_intersectbits(V[i], V[j]) * (len(clusters[V[i]]) +  len(clusters[V[j]]))
                GA[j][i] = GA[i][j]
                E.append((GA[i][j], i, j))
                Esum += GA[i][j]
    for i in range(len(V)):
        V_weight.append([i, 0])
        for j in range(len(V)):
            V_weight[i][1] += GA[i][j]
    V_weight = sorted(V_weight, reverse=True, key=lambda vw : vw[1])
    E = sorted(E, reverse=True, key=lambda ew : ew[0])
    #print(V)
    #print(V_weight)
    #print(E)
    #print(GA)
    ### step3: cut max-k E
    cutset = []
    vis_V = set()
    if len(E) == 0:
        cutset.append([0])
        vis_V.add(0)
    for i in range(min(MAXCUT_K, len(E))):
        if E[i][0] < 1:
            break
        for j in range(1, 3):
            if [E[i][j]] not in cutset:
                cutset.append([E[i][j]])
                vis_V.add(E[i][j])
    ### step4: cut the graph left
    E_sum_opt = 0
    for i in V_weight:
        if i[0] not in vis_V:
            opt_j = None
            opt = None
            for j in range(len(cutset)):
                cost = 0
                for k in cutset[j]:
                    cost += GA[i[0]][k]
                if opt is None or cost < opt:
                    opt_j = j
                    opt = cost
            cutset[opt_j].append(i[0])
            vis_V.add(i[0])
            E_sum_opt += opt
    #print(cutset)
    ### step5: final
    #r_score, r_clusters, r_centers
    r_clusters = [None] * len(workload)
    r_centers = []
    maxcut_point = []
    for i, c in enumerate(cutset):
        maxcut_point.append({})
        r_centers.append(qsplit_train_cluster_encoder(scope, [V[j] for j in c]))
        for j in c:
            maxcut_point[-1][V[j]] = len(clusters[V[j]])
            for k in clusters[V[j]]:
                r_clusters[k] = i
    r_score = (Esum - E_sum_opt) / max(1, Esum)
    #print(clusters)
   # maxcut_point = {i: len(clusters[i]) for i in clusters}
    print(r_score, Esum, E_sum_opt, len(cutset))
    #print(r_centers)
    #exit(-1)
    
    return r_score, r_clusters, r_centers, maxcut_point

# Clustering
def preproc_queries(workload, scope):
    all_scope = set(range(workload.shape[1]))
    not_scope = list(all_scope - set(scope))
    queries = (workload[:,:,0]!=-np.inf) | (workload[:,:,1]!=np.inf)
    queries = queries.astype(int)
    queries[:, not_scope] = 0
    return queries

def get_split_queries_Kmeans(n_clusters=2, seed=17):

    def split_queries_Kmeans(local_workload, scope, return_clusters=False):
        queries = preproc_queries(local_workload, scope)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        clusters = kmeans.fit_predict(queries)
        centers = kmeans.cluster_centers_

        if return_clusters:
            return queries, clusters, centers
        return split_queries_by_clusters(local_workload, clusters, scope, centers)

    return split_queries_Kmeans

