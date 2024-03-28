import tgx
#! use assert to ensure https://docs.pytest.org/en/8.0.x/getting-started.html

def check_is_sorted(test_list: list) -> bool:
    r"""
    helper function to check if a list is sorted in ascending order
    """
    is_sorted = all(a <= b for a, b in zip(test_list, test_list[1:]))
    return is_sorted


def count_snapshot_edges(snapshots: dict) -> int:
    r"""
    helper function to count the number of edges in a snapshot
    """
    num_edges = 0
    for snapshot_id, snapshot in snapshots.items():
        num_edges += snapshot.shape[1]
    return num_edges


# """
# This is a unit test to ensure that the conversion between CTDG and DTDG is correct and the timestamps are mapped correctly
# see pytest https://docs.pytest.org/en/8.0.x/
# """
# def test_custom_data_loading():
#     r"""
#     make sure discretization behavior in TGX is as expected
#     """
#     print ("hi")


def test_dtdg_loading():
    r"""
    1. load directly from TGX, count # of edges, count # of nodes
    2. compare with discretized version, count # of edges, count # of nodes
    3. check that the undirected graph used for HTGN matches with the original
    4. test that all negative samples are extracted for validation set
    """
    time_scale = "monthly"
    dataset = tgx.builtin.enron()
    ctdg = tgx.Graph(dataset)
    ctdg_num_edges = ctdg.number_of_edges()
    dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
    dtdg_num_edges = dtdg.number_of_edges()

    snapshot_id_list = list(dtdg.data.keys())
    
    assert ctdg_num_edges >= dtdg_num_edges, "Number of edges less or equal after discretization"
    assert check_is_sorted(ts_list) == True, "Timestamps are sorted in ascending order"
    assert check_is_sorted(snapshot_id_list) == True, "Snapshot IDs are sorted in ascending order"

    #* loading for DTDG methods on DTDG data
    from utils.data_util import loader

    data = loader(dataset="enron", time_scale="monthly")
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']

    train_snapshot_ids = list(train_data['edge_index'].keys())
    val_snapshot_ids = list(val_data['edge_index'].keys())
    test_snapshot_ids = list(test_data['edge_index'].keys())

    assert check_is_sorted(train_snapshot_ids) == True, "Train Snapshot IDs are sorted in ascending order"
    assert check_is_sorted(val_snapshot_ids) == True, "Val Snapshot IDs are sorted in ascending order"
    assert check_is_sorted(test_snapshot_ids) == True, "Test Snapshot IDs are sorted in ascending order"

    all_snapshot_ids = train_snapshot_ids + val_snapshot_ids + test_snapshot_ids

    assert all_snapshot_ids == snapshot_id_list, "All snapshot IDs are the same"

    num_edges = 0
    num_edges += count_snapshot_edges(train_data['original_edges'])
    num_edges += count_snapshot_edges(val_data['original_edges'])
    num_edges += count_snapshot_edges(test_data['original_edges'])

    assert num_edges == dtdg_num_edges, "number of edges sum to the same as raw data from train, val, test partitions"



# def test_ctdg_loading():
#     r"""
#     1. load directly from TGB, check # of edges
#     2. load TGB from TGX, check # of edges
#     3. discretize TGB dataset edges, check # of edges
#     """
#     print ("hi")



if __name__ == '__main__':
    test_dtdg_loading()






