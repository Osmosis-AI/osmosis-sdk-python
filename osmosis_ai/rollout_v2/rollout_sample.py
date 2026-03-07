from abc import ABC, abstractmethod

from osmosis_ai.rollout_v2.types import RolloutSample


class RolloutSampleSource(ABC):
    @abstractmethod
    def make_rollout_sample(self) -> RolloutSample:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> list:
        raise NotImplementedError
