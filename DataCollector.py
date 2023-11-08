from typing import Callable

import pandas


class DataCollector:
    def register_receive(self, on_collected_data: Callable[[pandas.DataFrame], None]):
        pass

    def start_collect(self) -> None:
        pass
