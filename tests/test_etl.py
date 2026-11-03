from etl import get_folder, data_frame_pattern


def test_pasta_vazia():
    assert bool(get_folder()) is True

def test_wrong_dataframe():
    assert not data_frame_pattern().empty