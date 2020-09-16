from tqdm import tqdm
import pickle
import math
import random
import pandas as pd
import math

data_dir = "data/"


def save_obj(obj, name):
    with open(data_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(data_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# load data
# sample_data = load_obj("SBdata2")
BasicFeatures = load_obj("pre_features_v2")
pre_features = BasicFeatures
edges_of_all_test_nodes_related = load_obj('edges_of_all_test_nodes_related')

# Function
# Salton Similarity
def salton_similarity(node1, node2):
    n1 = pre_features[node1]
    n2 = pre_features[node2]
    common_neighors = list(set(n1[2]).intersection(n2[2]))
    inter = len(common_neighors)
    degree_out_flow = n1[6]
    degree_in_flow = n2[4]

    if inter == 0:
        return 0
    else:
        try:
            sqrt_of_degree = math.sqrt(degree_out_flow * degree_in_flow)
            salton = inter / sqrt_of_degree
            probability = 1 / (1 - math.log(salton) * 0.2)
            return probability
        except:
            return 0


# Cosine
def Cosine(Node1, Node2):
    n1 = pre_features[Node1]
    n2 = pre_features[Node2]
    common_neighors = list(set(n1[2]).intersection(n2[2]))
    lm = len(common_neighors)
    if lm == 0:
        return 0
    else:
        return (0.0 + lm) / (len(n1[2]) * len(n2[2]))


def get_jaccard_coefficient(source, sink):
    """
    in: source::Node object
    in: sink::Node object
    return: jaccard's cofficient::numeric
    """
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]

    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    union_neighbours = neigbours_set_of_source | neigbours_set_of_sink
    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
    if len(union_neighbours) == 0:
        return 0.0
    return (len(common_neighbours) / len(union_neighbours))


def get_preferential_attachment(source, sink):
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]

    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)

    return len(neigbours_set_of_source) * len(neigbours_set_of_sink)


def get_adamic_adar(source, sink):
    # transform
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]

    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)
    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
    # get the summation
    score = 0
    for common_node in common_neighbours:
        if math.log(len(BasicFeatures[common_node][2])) == 0:
            return 0.0
        score = score + 1 / math.log(len(BasicFeatures[common_node][2]))
    return score


def get_resource_allocation(source, sink):
    neighbours_of_source_list = BasicFeatures[source][2]
    neighbours_of_sink_list = BasicFeatures[sink][2]
    #     print(neighbours_of_source_list)
    #     print(neighbours_of_sink_list)
    neigbours_set_of_source = set(neighbours_of_source_list)
    neigbours_set_of_sink = set(neighbours_of_sink_list)

    common_neighbours = neigbours_set_of_source & neigbours_set_of_sink
    #     print(common_neighbours)
    score = 0
    for common_node in common_neighbours:
        # number of the neighbours of the common_node
        try:
            single_common_node_score = 1 / BasicFeatures[common_node][0]
        except:
            single_common_node_score = 0
        score = score + single_common_node_score
    return score


# how similar are the outbound neighbors of source to sink
# either JA, PA, AA
def get_outbound_similarity_score(source, sink, metric):
    # get the outbound_node of source
    outbound_node_for_source_set = set(BasicFeatures[source][5])
    summation = 0
    for outbound_node_for_source in outbound_node_for_source_set:
        summation = summation + metric(sink, outbound_node_for_source)
    if len(outbound_node_for_source_set) == 0:
        return 0
    score = 1 / len(outbound_node_for_source_set) * summation
    return score


# either JA, PA, AA
def get_inbound_similarity_score(source, sink, metric):
    # get the inbound_node of sink
    inbound_node_for_sink_set = set(BasicFeatures[source][3])
    summation = 0
    for inbound_node_for_sink in inbound_node_for_sink_set:
        summation = summation + metric(source, inbound_node_for_sink)
    if len(inbound_node_for_sink_set) == 0:
        return 0
    score = 1 / len(inbound_node_for_sink_set) * summation
    return score


def get_common_neighbours(node1, node2):
    try:
        n1 = pre_features[node1]
        n2 = pre_features[node2]
        common_neighors = list(set(n1[2]).intersection(n2[2]))
        return common_neighors
    except:
        return 0


def gen_training_df(final_edges):
    training_df = pd.DataFrame()
    for edge in tqdm(final_edges):
        source = edge[0]
        sink = edge[1]
        label = edge[2]
        salton_similarity_score = salton_similarity(source, sink)
        cosine = Cosine(source, sink)
        jaccard_coefficient = get_jaccard_coefficient(source, sink)
        preferential_attachment = get_preferential_attachment(source, sink)
        adamic_adar = get_adamic_adar(source, sink)
        resource_allocation = get_resource_allocation(source, sink)

        salton_similarity_score_out = get_outbound_similarity_score(source, sink, salton_similarity)
        cosine_out = get_outbound_similarity_score(source, sink, Cosine)
        jaccard_coefficient_out = get_outbound_similarity_score(source, sink, get_jaccard_coefficient)
        preferential_attachment_out = get_outbound_similarity_score(source, sink, get_preferential_attachment)
        adamic_adar_out = get_outbound_similarity_score(source, sink, get_adamic_adar)
        resource_allocation_out = get_outbound_similarity_score(source, sink, get_resource_allocation)

        salton_similarity_score_in = get_inbound_similarity_score(source, sink, salton_similarity)
        cosine_in = get_inbound_similarity_score(source, sink, Cosine)
        jaccard_coefficient_in = get_inbound_similarity_score(source, sink, get_jaccard_coefficient)
        preferential_attachment_in = get_inbound_similarity_score(source, sink, get_preferential_attachment)
        adamic_adar_in = get_inbound_similarity_score(source, sink, get_adamic_adar)
        resource_allocation_in = get_inbound_similarity_score(source, sink, get_resource_allocation)

        df_row = pd.DataFrame([source, sink, label,
                               salton_similarity_score,
                               cosine,
                               jaccard_coefficient,
                               preferential_attachment,
                               adamic_adar,
                               resource_allocation,
                               salton_similarity_score_out,
                               cosine_out,
                               jaccard_coefficient_out,
                               preferential_attachment_out,
                               adamic_adar_out,
                               resource_allocation_out,
                               salton_similarity_score_in,
                               cosine_in,
                               jaccard_coefficient_in,
                               preferential_attachment_in,
                               adamic_adar_in,
                               resource_allocation_in
                               ]).T
        training_df = training_df.append(df_row)
    return training_df


def gen_training_edges(num_of_edges):
    """
    num_of_edges: number of positive edges to generate
    """
    # generate the positive_edge
    ps_edges = random.sample(edges_of_all_test_nodes_related, num_of_edges)
    ps_edges_set = set(ps_edges)  # Here we change ps_edges to hash set to decrease the time complexity
    nodes = set()
    for edge in tqdm(ps_edges):
        nodes.add(edge[0])
        nodes.add(edge[1])
    # generate the negative edges
    count = 0
    final_edges = list()
    while count < num_of_edges:
        if count % 1000 == 0:
            print(count)
        node1, node2 = random.sample(nodes, 2)
        if (node1, node2) not in ps_edges_set:
            count += 1
            final_edges.append((node1, node2, 0))
    for edge in ps_edges:
        final_edges.append((edge[0], edge[1], 1))
    return final_edges