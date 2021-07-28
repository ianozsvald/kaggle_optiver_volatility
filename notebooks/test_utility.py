from utility import make_unique_time_ids, get_training_stock_ids


def test_make_unique_time_ids():
    """Check we get a 25/75% split on ides and that they don't overlap"""
    all_time_ids = range(100)
    time_ids_train, time_ids_test = make_unique_time_ids(all_time_ids, test_size=0.25)
    assert len(time_ids_train) + len(time_ids_test) == len(all_time_ids)
    assert len(time_ids_train.intersection(time_ids_test)) == 0

    
def test_fetching_stock_ids():
    id_list = get_training_stock_ids()
    assert 1 in id_list
    assert 300 not in id_list
    assert 12 not in id_list, "for whatever reason stock id 12 doesn't exist"
    assert len(id_list) > 0