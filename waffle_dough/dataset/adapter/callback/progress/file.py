from pathlib import Path

from waffle_utils.file import io

from .progress import DatasetAdapterProgressCallback


class DatasetAdapterFileProgressCallback(DatasetAdapterProgressCallback):
    def __init__(self, file, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file = file

    @property
    def file(self) -> Path:
        return self._file

    @file.setter
    def file(self, file: str):
        self._file = Path(file).absolute()

    def on_loop_start(self, total_steps: int):
        io.make_directory(self.file.parent)
        io.save_json(self.get_progress_info().to_dict(), self.file)

    def on_step_start(self):
        io.save_json(self.get_progress_info().to_dict(), self.file)

    def on_step_end(self, current_step: int = None):
        io.save_json(self.get_progress_info().to_dict(), self.file)

    def on_loop_end(self):
        io.save_json(self.get_progress_info().to_dict(), self.file)
