import pytest

from waffle_dough.callback.dataset.adapter.progress import (
    DatasetAdapterProgressCallback,
    DatasetAdapterTqdmProgressCallback,
)
from waffle_dough.dataset.adapter.base_adapter import BaseAdapter


def test_dataset_adapter_progress_callback():
    callback = DatasetAdapterProgressCallback()
    adapter = BaseAdapter(callbacks=[callback])

    assert callback.current_step == None
    assert callback.total_steps == None
    assert callback.start_time == None

    adapter.run_hook("on_loop_start", 10)
    assert callback.current_step == 0
    assert callback.total_steps == 10
    assert callback.start_time != None

    adapter.run_hook("on_step_end", 1)
    assert callback.current_step == 1
    assert callback.total_steps == 10
    assert callback.start_time != None

    adapter.run_hook("on_step_end", 2)
    assert callback.current_step == 2
    assert callback.total_steps == 10
    assert callback.start_time != None

    with pytest.raises(TypeError):
        adapter.run_hook("on_step_end")

    with pytest.raises(ValueError):
        adapter.run_hook("on_step_end", 0)

    with pytest.raises(ValueError):
        adapter.run_hook("on_step_end", 11)

    with pytest.raises(ValueError):
        adapter.run_hook("on_loop_start", 20)

    adapter.run_hook("on_loop_end")
    assert callback.current_step == 10
    assert callback.total_steps == 10
    assert callback.get_remaining_time() == 0.0


def test_dataset_adapter_tqdm_progress_callback():
    callback = DatasetAdapterTqdmProgressCallback(desc="test")
    adapter = BaseAdapter(callbacks=[callback])

    assert callback.tqdm_bar is None

    adapter.run_hook("on_loop_start", 10)
    assert callback.tqdm_bar is not None

    adapter.run_hook("on_step_end", 1)
    assert callback.tqdm_bar is not None

    adapter.run_hook("on_step_end", 2)
    assert callback.tqdm_bar is not None

    adapter.run_hook("on_loop_end")
    assert callback.tqdm_bar is None


def test_dataset_adapter_progress_callback_multi():
    callback1 = DatasetAdapterProgressCallback()
    callback2 = DatasetAdapterTqdmProgressCallback()
    adapter = BaseAdapter(callbacks=[callback1, callback2])

    adapter.run_hook("on_loop_start", 10)
    assert callback1.current_step == callback2.current_step == 0

    adapter.run_hook("on_step_end", 1)
    assert callback1.current_step == callback2.current_step == 1
    assert callback1.get_remaining_time() == callback2.get_remaining_time()

    adapter.run_hook("on_step_end", 2)
    assert callback1.current_step == callback2.current_step == 2
    assert callback1.get_remaining_time() == callback2.get_remaining_time()

    adapter.run_hook("on_loop_end")
    assert callback1.current_step == callback2.current_step == 10
    assert callback1.get_remaining_time() == callback2.get_remaining_time()
