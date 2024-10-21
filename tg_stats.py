
import numpy as np
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from utils.data_util import loader, load_dtdg
from utils.utils_func import generate_splits, convert_to_torch_extended, set_random, get_snapshot_batches


# ============ Helper Functions
def get_edge_indices_of_snapshot(snapshot_idx, ts_map, max_ts_split, data_timestamps):
    """
    returns indices of the edges of the snapshot
    :param snapshot_idx: snapshot index
    :param ts_map: mapping of the snapshot indices to actual timestamps
    :param max_ts_split: maximum timestamps of the data split
    :param data_timestamps: edge timestamps
    """
    max_snapshot_idx = max(list(ts_map.keys()))
    if snapshot_idx > max_snapshot_idx:
        raise ValueError(
            f"Snapshot index is greater than max snapshot index!\n\tmax_snapshot_idx: {max_snapshot_idx}, snapshot_idx: {snapshot_idx}")
    if snapshot_idx != max_snapshot_idx:
        indices = (data_timestamps >= ts_map[snapshot_idx]) & (
            data_timestamps < ts_map[snapshot_idx + 1])
    else:
        indices = (data_timestamps >= ts_map[snapshot_idx]) & (
            data_timestamps <= max_ts_split)

    return indices


def remove_duplicate_edges(srcs: list, dsts: list, tss: list, msgs: list, unique_snapshot_idx: int):
    """
    remove duplicate edges from a snapshot
    add the snapshot index as the timestamps
    :param srcs: sources (list)
    :param dsts: destinations (list)
    :param tss: timestamps (list)
    :param msgs: edge messages (list)
    :param unique_snapshot_idx: snapshot index (int)
    """
    unique_edge_srcs, unique_edge_dsts, unique_edge_timestamps, unique_edge_msgs, snapshot_idxs = [], [], [], [], []
    unique_edge_dict = {}
    for src, dst, ts, msg in zip(srcs, dsts, tss, msgs):
        if (src, dst) not in unique_edge_dict:
            unique_edge_dict[(src, dst)] = 1
            unique_edge_srcs.append(src)
            unique_edge_dsts.append(dst)
            unique_edge_timestamps.append(ts)
            unique_edge_msgs.append(msg)
            # same value for all edges of a snapshot
            snapshot_idxs.append(unique_snapshot_idx)
    return np.array(unique_edge_srcs), np.array(unique_edge_dsts), \
        np.array(unique_edge_timestamps), np.array(
            unique_edge_msgs), np.array(snapshot_idxs)


def get_split_snapshots(ts_map, max_ts_split, full_data, edge_indices):
    """
    get the snapshots from a data split
    :param ts_map: timestamp map
    :param max_split_ts: max timestamp of the split
    :param full_data: TGB full_data
    :param edge_indices: edge indices for full_data
    """
    split_snapshots = {}
    for snapshot_idx in list(ts_map.keys()):
        snapshot_edge_indices = get_edge_indices_of_snapshot(
            snapshot_idx, ts_map, max_ts_split, full_data.t)
        sources, destinations, timestamps, msgs, snapshot_idxs = remove_duplicate_edges(
            full_data.src[snapshot_edge_indices],
            full_data.dst[snapshot_edge_indices],
            full_data.t[snapshot_edge_indices],
            full_data.msg[snapshot_edge_indices],
            snapshot_idx,
        )
        split_snapshots[snapshot_idx] = {
            'sources': sources,
            'destinations': destinations,
            'timestamps': timestamps,
            'msg': msgs,
            'e_idx': np.array(edge_indices[snapshot_edge_indices]),
            'snapshot_idx': snapshot_idxs,
        }
    return split_snapshots

# ===================================


def get_stats_CTDG(dataset):
    print("="*100)
    print(f"DATA: {dataset}")
    print("="*100)

    # ctdg dataset
    dataset = PyGLinkPropPredDataset(dataset, root="datasets")
    full_data = dataset.get_TemporalData()

    num_nodes_CT = len(set(np.concatenate((full_data.src, full_data.dst))))
    num_edges_CT = len(full_data.src)
    num_uniq_ts_CT = len(set(np.array(full_data.t)))
    print(f"INFO: CDTDG: num. nodes: {num_nodes_CT}")
    print(f"INFO: CDTDG: num. edges: {num_edges_CT}")
    print(f"INFO: CDTDG: num. uniq. timestamps: {num_uniq_ts_CT}")

    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    train_edges = full_data[train_mask]
    val_edges = full_data[val_mask]
    test_edges = full_data[test_mask]

    # compute unique edges
    train_val_uniq_edges = {}
    for src, dst in zip(np.array(train_edges.src), np.array(train_edges.dst)):
        if (src, dst) not in train_val_uniq_edges:
            train_val_uniq_edges[(src, dst)] = 1
    for src, dst in zip(np.array(val_edges.src), np.array(val_edges.dst)):
        if (src, dst) not in train_val_uniq_edges:
            train_val_uniq_edges[(src, dst)] = 1

    test_uniq_edges = {}
    for src, dst in zip(np.array(test_edges.src), np.array(test_edges.dst)):
        if (src, dst) not in test_uniq_edges:
            test_uniq_edges[(src, dst)] = 1

    # compute `surprise`
    difference = 0
    for e in test_uniq_edges:
        if e not in train_val_uniq_edges:
            difference += 1
    surprise_CT = float(difference * 1.0 / len(np.array(test_edges.src)))
    print(f"INFO: CDTDG: surprise: {surprise_CT}")

    # compute `num_unique_edges`
    uniq_edges = train_val_uniq_edges
    for src, dst in zip(np.array(test_edges.src), np.array(test_edges.dst)):
        if (src, dst) not in uniq_edges:
            uniq_edges[(src, dst)] = 1
    print(f"INFO: CTDG: Total Number of Unique Edges: {len(uniq_edges)}")
    print("="*100)


def get_stats_CTDG_discretized(dataset_name, time_scale):

    print("="*100)
    print(f"DATA: {dataset_name}, TIME_SCALE: {time_scale}")
    print("="*100)

    # === TGB dataloading
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
    dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
    full_data = dataset.get_TemporalData()
    edge_ids = np.arange(len(np.array(full_data.src)))

    # get masks
    train_data = full_data[dataset.train_mask]
    val_data = full_data[dataset.val_mask]
    test_data = full_data[dataset.test_mask]

    # ======================
    # === discretize dataset
    data = loader(dataset_name, time_scale)
    train_ts = data['train_data']['ts_map']
    val_ts = data['val_data']['ts_map']
    test_ts = data['test_data']['ts_map']

    train_snapshots = get_split_snapshots(
        train_ts, max(train_data.t), full_data, edge_ids)
    val_snapshots = get_split_snapshots(
        val_ts, max(val_data.t), full_data, edge_ids)
    test_snapshots = get_split_snapshots(
        test_ts, max(test_data.t), full_data, edge_ids)

    num_snapshot_train = len(train_snapshots)
    num_snapshot_val = len(val_snapshots)
    num_snapshot_test = len(test_snapshots)
    num_total_snapshots = num_snapshot_train + num_snapshot_val + num_snapshot_test
    print(
        f"INFO: CTDG: Number of snapshots: Total: {num_total_snapshots}, Train: {num_snapshot_train}, Val: {num_snapshot_val}, Test: {num_snapshot_test}")

    def get_unique_edges(split_snapshots):
        uniq_edge = {}
        num_edges = 0
        for snapshot in split_snapshots:
            num_edges += len(split_snapshots[snapshot]['sources'])
            for src, dst in zip(split_snapshots[snapshot]['sources'], split_snapshots[snapshot]['destinations']):
                if (src, dst) not in uniq_edge:
                    uniq_edge[(src, dst)] = 1
        return uniq_edge, num_edges

    uniq_edge_train, num_edges_train = get_unique_edges(train_snapshots)
    uniq_edge_val, num_edges_val = get_unique_edges(val_snapshots)
    uniq_edge_test, num_edges_test = get_unique_edges(test_snapshots)
    total_num_edges = num_edges_train + num_edges_val + num_edges_test

    print(
        f"INFO: CTDG: Number of Edges: Total: {total_num_edges}, Train: {num_edges_train}, Val: {num_edges_val}, Test: {num_edges_test}")
    print(
        f"INFO: CTDG: Number of Edges: Unique: Train: {len(uniq_edge_train)}, Val: {len(uniq_edge_val)}, Test: {len(uniq_edge_test)}")

    # compute `surprise`
    uniq_edges_train_val = uniq_edge_train
    for e in uniq_edge_val:
        if e not in uniq_edges_train_val:
            uniq_edges_train_val[e] = 1

    difference = 0
    for e in uniq_edge_test:
        if e not in uniq_edges_train_val:
            difference += 1
    surprise_DT = float(difference * 1.0 / (num_edges_test * 1.0))
    print(f"INFO: CTDG: Surprise (after Discretization): {surprise_DT}")

    # compute `moving reocurrence`
    def get_uniq_edges_src_dst_list(srcs, dsts):
        uniq_edges = {}
        for src, dst in zip(srcs, dsts):
            if (src, dst) not in uniq_edges:
                uniq_edges[(src, dst)] = 1
        return uniq_edges

    sum_reocurrence = 0.0
    all_snapshots = {**train_snapshots, **val_snapshots, **test_snapshots}

    prev_snapshot = None
    zero_edge_snapshot_count = 0
    for ii, snapshot in enumerate(all_snapshots):
        if ii != 0:
            # current snapshot
            curr_src_nodes = all_snapshots[snapshot]['sources']
            curr_dst_nodes = all_snapshots[snapshot]['destinations']
            curr_uniq_edges = get_uniq_edges_src_dst_list(
                curr_src_nodes, curr_dst_nodes)
            curr_num_uniqu_edges = len(curr_uniq_edges)

            if len(curr_src_nodes) == 0:
                zero_edge_snapshot_count += 1

            # previous snapshot
            prev_src_nodes = all_snapshots[prev_snapshot]['sources']
            prev_dst_nodes = all_snapshots[prev_snapshot]['destinations']
            prev_uniq_edges = get_uniq_edges_src_dst_list(
                prev_src_nodes, prev_dst_nodes)

            # compute intersection
            curr_prev_intersect = {
                key: curr_uniq_edges[key] for key in curr_uniq_edges if key in prev_uniq_edges}

            if curr_num_uniqu_edges != 0:
                current_reocurrence = (
                    len(curr_prev_intersect) * 1.0) / (curr_num_uniqu_edges * 1.0)
                sum_reocurrence += current_reocurrence

        prev_snapshot = snapshot

    moving_reocurrence = (sum_reocurrence * 1.0) / \
        (len(all_snapshots) - zero_edge_snapshot_count * 1.0)
    print(f"INFO: CTDG: Moving Reocurrence: {moving_reocurrence}")


def get_stats_DTDG(dataset_name, time_scale):
    print("="*100)
    print(f"DATA: {dataset_name}, TIME_SCALE: {time_scale}")
    print("="*100)

    dtdg, ts_list = load_dtdg(dataset_name, time_scale)

    full_data = dtdg.export_full_data()
    src_node_ids = full_data["sources"]
    dst_node_ids = full_data["destinations"]
    node_interact_times = full_data["timestamps"]
    edge_ids = np.arange(len(src_node_ids))  # required by NAT

    num_nodes_CT = len(set(np.concatenate((src_node_ids, dst_node_ids))))
    num_edges_CT = len(np.array(src_node_ids))
    num_uniq_ts_CT = len(set(np.array(node_interact_times)))
    print(f"INFO: DTDG: num. nodes: {num_nodes_CT}")
    print(f"INFO: DTDG: num. edges: {num_edges_CT}")
    print(f"INFO: DTDG: num. uniq. timestamps: {num_uniq_ts_CT}")

    # get masks
    # get a list of snapshot batches from the timestamps
    snapshot_indices = get_snapshot_batches(node_interact_times)
    train_mask, val_mask, test_mask = generate_splits(full_data,
                                                      val_ratio=0.15,
                                                      test_ratio=0.15,
                                                      )

    # compute unique edges
    train_val_uniq_edges = {}
    for src, dst in zip(np.array(full_data['sources'][train_mask]), np.array(full_data['destinations'][train_mask])):
        if (src, dst) not in train_val_uniq_edges:
            train_val_uniq_edges[(src, dst)] = 1
    for src, dst in zip(np.array(full_data['sources'][val_mask]), np.array(full_data['destinations'][val_mask])):
        if (src, dst) not in train_val_uniq_edges:
            train_val_uniq_edges[(src, dst)] = 1

    test_uniq_edges = {}
    for src, dst in zip(np.array(full_data['sources'][test_mask]), np.array(full_data['destinations'][test_mask])):
        if (src, dst) not in test_uniq_edges:
            test_uniq_edges[(src, dst)] = 1

    # compute `surprise`
    difference = 0
    for e in test_uniq_edges:
        if e not in train_val_uniq_edges:
            difference += 1
    surprise_CT = float(difference * 1.0 /
                        len(np.array(full_data['sources'][test_mask])))
    print(f"INFO: DTDG: surprise: {surprise_CT}")

    # ===========================================
    print("-"*50)
    start_times = {
        'train': node_interact_times[train_mask][0].item(),
        'val': node_interact_times[val_mask][0].item(),
        'test': node_interact_times[test_mask][0].item(),
    }

    end_times = {
        'train': node_interact_times[train_mask][-1].item(),
        'val': node_interact_times[val_mask][-1].item(),
        'test': node_interact_times[test_mask][-1].item(),
    }

    def get_unique_edges(split_snapshot_indices):
        uniqe_edges = {}
        num_edges = 0
        for snap_idx in split_snapshot_indices:
            idx_start = snapshot_indices[snap_idx][0]
            idx_end = snapshot_indices[snap_idx][1]

            src_nodes = src_node_ids[idx_start:idx_end]
            dst_nodes = dst_node_ids[idx_start:idx_end]

            num_edges += len(src_nodes)

            for src, dst in zip(src_nodes, dst_nodes):
                if (src, dst) not in uniqe_edges:
                    uniqe_edges[(src, dst)] = 1

        return uniqe_edges, num_edges

    train_snapshot_indices = range(
        start_times['train'], end_times['train'] + 1)
    train_uniq_edges, train_num_edges = get_unique_edges(
        train_snapshot_indices)

    val_snapshot_indices = range(start_times['val'], end_times['val'] + 1)
    val_uniq_edges, val_num_edges = get_unique_edges(val_snapshot_indices)

    test_snapshot_indices = range(start_times['test'], end_times['test'] + 1)
    test_uniq_edges, test_num_edges = get_unique_edges(test_snapshot_indices)

    total_num_edges = train_num_edges + val_num_edges + test_num_edges

    print(
        f"INFO: DTDG: Number of Edges: Total: {total_num_edges}, Train: {train_num_edges}, Val: {val_num_edges}, Test: {test_num_edges}")
    print(
        f"INFO: DTDG: Number of Edges: Unique: Train: {len(train_uniq_edges)}, Val: {len(val_uniq_edges)}, Test: {len(test_uniq_edges)}")

    # compute `surprise`
    uniq_edges_train_val = train_uniq_edges
    for e in val_uniq_edges:
        if e not in uniq_edges_train_val:
            uniq_edges_train_val[e] = 1

    difference = 0
    for e in test_uniq_edges:
        if e not in uniq_edges_train_val:
            difference += 1
    surprise_DT = float(difference * 1.0 / (test_num_edges * 1.0))
    print(f"INFO: DTDG: Surprise (after Discretization): {surprise_DT}")

    # compute `moving reocurrence`
    def get_uniq_edges_src_dst_list(srcs, dsts):
        uniq_edges = {}
        for src, dst in zip(srcs, dsts):
            if (src, dst) not in uniq_edges:
                uniq_edges[(src, dst)] = 1
        return uniq_edges

    sum_reocurrence = 0.0
    zero_edge_snapshot_count = 0
    snapshot_indices_keys = list(snapshot_indices.keys())
    for ii, snap_idx in enumerate(snapshot_indices_keys):
        if ii != 0:
            # current snapshot
            curr_snap_start = snapshot_indices[snap_idx][0]
            curr_snap_end = snapshot_indices[snap_idx][1]

            curr_src_nodes = src_node_ids[curr_snap_start: curr_snap_end]
            curr_dst_nodes = dst_node_ids[curr_snap_start: curr_snap_end]

            if len(curr_src_nodes) == 0:
                zero_edge_snapshot_count += 1

            curr_uniq_edges = get_uniq_edges_src_dst_list(
                curr_src_nodes, curr_dst_nodes)
            curr_num_uniqu_edges = len(curr_uniq_edges)

            # previous snapshot
            prev_snap_start = snapshot_indices[snapshot_indices_keys[ii - 1]][0]
            prev_snap_end = snapshot_indices[snapshot_indices_keys[ii - 1]][1]

            prev_src_nodes = src_node_ids[prev_snap_start: prev_snap_end]
            prev_dst_nodes = dst_node_ids[prev_snap_start: prev_snap_end]

            prev_uniq_edges = get_uniq_edges_src_dst_list(
                prev_src_nodes, prev_dst_nodes)

            # compute intersection
            curr_prev_intersect = {
                key: curr_uniq_edges[key] for key in curr_uniq_edges if key in prev_uniq_edges}
            current_reocurrence = (
                len(curr_prev_intersect) * 1.0) / (curr_num_uniqu_edges * 1.0)
            sum_reocurrence += current_reocurrence
    moving_reocurrence = (sum_reocurrence * 1.0) / \
        (len(snapshot_indices_keys) - zero_edge_snapshot_count * 1.0)
    print(f"INFO: DTDG: Moving Reocurrence: {moving_reocurrence}")


def main():
    # ------------------------
    # ------------------------ CTDG
    # ------------------------
    # dataset_name = 'tgbl-wiki'
    # time_scale = 'hourly'

    # dataset_name = 'tgbl-review'
    # time_scale = 'monthly'

    # dataset_name = 'tgbl-subreddit'
    # time_scale = 'hourly'

    # dataset_name = 'tgbl-lastfm'
    # time_scale = 'weekly'

    # get_stats_CTDG(dataset_name)
    # get_stats_CTDG_discretized(dataset_name, time_scale)

    # ------------------------
    # ------------------------ DTDG
    # ------------------------

    # # uci
    # dataset_name = "uci"
    # time_scale = "weekly"

    # # enron
    # dataset_name = "enron"
    # time_scale = "monthly"

    # # contacts
    # dataset_name = "contacts"
    # time_scale = "hourly"

    # social_evo
    dataset_name = "social_evo"
    time_scale = "daily"

    # # mooc
    # dataset_name = "mooc"
    # time_scale = "daily"

    # canparl
    # dataset_name="canparl"
    # time_scale="biyearly"

    get_stats_DTDG(dataset_name, time_scale)


if __name__ == '__main__':
    main()
