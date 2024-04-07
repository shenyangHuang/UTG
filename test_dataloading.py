import tgx
import torch
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
    """

    #* load directly from TGX
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


    #* checked exported edgelist
    full_data = dtdg.export_full_data()
    sources = full_data["sources"]
    destinations = full_data["destinations"]
    timestamps = full_data["timestamps"]
    assert len(sources) == len(destinations) == len(timestamps) == dtdg_num_edges, "Number of sources, destinations, timestamps are the same"

    #* construct snapshots based on timestamps
    from utils.utils_func import get_snapshot_batches

    index_dict = get_snapshot_batches(timestamps)
    assert len(snapshot_id_list) == len(index_dict), "same number of snapshots as index dict"


    #* check if splits are the same too
    from utils.utils_func import generate_splits
    val_ratio = 0.15
    test_ratio = 0.15
    train_mask, val_mask, test_mask = generate_splits(full_data,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_ts = timestamps[train_mask]
    val_ts = timestamps[val_mask]
    test_ts = timestamps[test_mask]


    #* check DTDG method dataloading setup
    from utils.data_util import loader

    data = loader(dataset="enron", time_scale="monthly")
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']

    train_snapshot_ids = list(train_data['original_edges'].keys())
    val_snapshot_ids = list(val_data['original_edges'].keys())
    test_snapshot_ids = list(test_data['original_edges'].keys())

    assert check_is_sorted(train_snapshot_ids) == True, "Train Snapshot IDs are sorted in ascending order"
    assert check_is_sorted(val_snapshot_ids) == True, "Val Snapshot IDs are sorted in ascending order"
    assert check_is_sorted(test_snapshot_ids) == True, "Test Snapshot IDs are sorted in ascending order"

    assert train_ts[0] == train_snapshot_ids[0], "First train snapshot ID is the same as the first timestamp"
    assert train_ts[-1] == train_snapshot_ids[-1], "Last train snapshot ID is the same as the last timestamp"
    
    print (val_ts[-1], val_snapshot_ids[-1])
    assert val_ts[0] == val_snapshot_ids[0], "First val snapshot ID is the same as the first timestamp"
    assert val_ts[-1] == val_snapshot_ids[-1], "last val snapshot ID is the same as the last timestamp"

    
    assert test_ts[0] == test_snapshot_ids[0], "First test snapshot ID is the same as the first timestamp"
    assert test_ts[-1] == test_snapshot_ids[-1], "Last test snapshot ID is the same as the last timestamp"
    all_snapshot_ids = train_snapshot_ids + val_snapshot_ids + test_snapshot_ids

    assert all_snapshot_ids == snapshot_id_list, "All snapshot IDs are the same"

    num_edges = 0
    num_edges += count_snapshot_edges(train_data['original_edges'])
    num_edges += count_snapshot_edges(val_data['original_edges'])
    num_edges += count_snapshot_edges(test_data['original_edges'])

    assert num_edges == dtdg_num_edges, "number of edges sum to the same as raw data from train, val, test partitions"




def test_ctdg_loading():
    r"""
    1. TGB dataloading
    2. UTG dataloading
    """
    DATA = "tgbl-wiki"
    time_scale = "hourly"


    from tgb.linkproppred.dataset import LinkPropPredDataset
    #* TGB dataloading
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
    data = dataset.full_data  
    metric = dataset.eval_metric


    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask





    from utils.data_util import loader
    data = loader(dataset=DATA, time_scale=time_scale)



if __name__ == '__main__':
    test_dtdg_loading()






