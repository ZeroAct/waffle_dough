import pytest

from waffle_dough.dataset.adapter.base_adapter import BaseAdapter
from waffle_dough.dataset.adapter.callback import (
    DatasetAdapterFileProgressCallback,
    DatasetAdapterProgressCallback,
    DatasetAdapterTqdmProgressCallback,
)


def test_dataset_adapter_progress_callback():
    adapter = BaseAdapter()

    assert DatasetAdapterProgressCallback._current_step == None
    assert DatasetAdapterProgressCallback._total_steps == None
    assert DatasetAdapterProgressCallback._start_time == None

    adapter.run_callback_hooks("on_loop_start", 10)
    assert DatasetAdapterProgressCallback._current_step == 0
    assert DatasetAdapterProgressCallback._total_steps == 10
    assert DatasetAdapterProgressCallback._start_time != None

    adapter.run_callback_hooks("on_step_end", 1)
    assert DatasetAdapterProgressCallback._current_step == 1
    assert DatasetAdapterProgressCallback._total_steps == 10
    assert DatasetAdapterProgressCallback._start_time != None

    adapter.run_callback_hooks("on_step_end", 2)
    assert DatasetAdapterProgressCallback._current_step == 2
    assert DatasetAdapterProgressCallback._total_steps == 10
    assert DatasetAdapterProgressCallback._start_time != None

    with pytest.raises(ValueError):
        adapter.run_callback_hooks("on_step_end", 0)

    with pytest.raises(ValueError):
        adapter.run_callback_hooks("on_step_end", 11)

    with pytest.raises(ValueError):
        adapter.run_callback_hooks("on_loop_start", 20)

    adapter.run_callback_hooks("on_loop_end")
    assert DatasetAdapterProgressCallback._current_step == 10
    assert DatasetAdapterProgressCallback._total_steps == 10


def test_dataset_adapter_tqdm_progress_callback():
    callback = DatasetAdapterTqdmProgressCallback(desc="test")
    adapter = BaseAdapter(callbacks=[callback])

    assert callback.tqdm_bar is None

    adapter.run_callback_hooks("on_loop_start", 10)
    assert callback.tqdm_bar is not None

    adapter.run_callback_hooks("on_step_end", 1)
    assert callback.tqdm_bar is not None

    adapter.run_callback_hooks("on_step_end", 2)
    assert callback.tqdm_bar is not None

    adapter.run_callback_hooks("on_loop_end")
    assert callback.tqdm_bar is None


def test_dataset_adapter_file_progress_callback(tmpdir):
    def read_json(file):
        import json

        with open(file) as f:
            return json.load(f)

    file = tmpdir.join("progress.json").strpath
    callback = DatasetAdapterFileProgressCallback(file)
    adapter = BaseAdapter(callbacks=[callback])

    adapter.run_callback_hooks("on_loop_start", 10)
    assert read_json(file)["total_steps"] == 10

    adapter.run_callback_hooks("on_step_end", 1)
    assert read_json(file)["current_step"] == 1

    adapter.run_callback_hooks("on_step_end", 2)
    assert read_json(file)["current_step"] == 2

    adapter.run_callback_hooks("on_loop_end")
    assert read_json(file)["current_step"] == 10


def test_dataset_adapter_progress_callback_multi(tmpdir):
    callback1 = DatasetAdapterFileProgressCallback(tmpdir.join("progress1.json").strpath)
    callback2 = DatasetAdapterTqdmProgressCallback()
    adapter = BaseAdapter(callbacks=[callback1, callback2])

    adapter.run_callback_hooks("on_loop_start", 10)
    assert callback1.current_step == callback2.current_step == 0

    adapter.run_callback_hooks("on_step_end", 1)
    assert callback1.current_step == callback2.current_step == 1
    assert callback1.get_remaining_time() == callback2.get_remaining_time()

    adapter.run_callback_hooks("on_step_end", 2)
    assert callback1.current_step == callback2.current_step == 2
    assert callback1.get_remaining_time() == callback2.get_remaining_time()

    adapter.run_callback_hooks("on_loop_end")
    assert callback1.current_step == callback2.current_step == 10
    assert callback1.get_remaining_time() == callback2.get_remaining_time()
