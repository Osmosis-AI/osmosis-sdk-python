from typing import Any, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from contextvars import ContextVar

from osmosis_ai.rollout_v2.types import (
    RolloutCompleteRequest,
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
    GraderCompleteRequest,
    GraderStatus,
    AgentWorkflowConfig,
)
from osmosis_ai.rollout_v2.rollout_sample import RolloutSampleSource

rollout_contextvar: ContextVar = ContextVar("rollout_contextvar", default=None)
TConfig = TypeVar("TConfig", bound=AgentWorkflowConfig)

@dataclass
class RolloutContext:
    rollout_id: str
    completion_callback_url: str
    chat_completions_url: str
    rollout_sample_sources: List[RolloutSampleSource] = field(default_factory=list)
    status: RolloutStatus = RolloutStatus.PENDING
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None

    def __enter__(self):
        rollout_contextvar.set(self)

    def __exit__(self, exc_type, exc_value, traceback):
        rollout_contextvar.set(None)

    def set_rollout_status(self, status: RolloutStatus):
        self.status = status

    def set_rollout_error(
        self, *, message: str, category: RolloutErrorCategory
    ) -> None:
        self.err_message = message
        self.err_category = category

    def make_rollout_complete_request(self) -> RolloutCompleteRequest:
        # TODO: in the future, maybe we want to extract some metadata from agents that we are using
        # samples = [source.make_rollout_sample() for source in self.rollout_sample_sources]
        return RolloutCompleteRequest(
            rollout_id=self.rollout_id,
            status=self.status,
            err_message=self.err_message,
            err_category=self.err_category,
        )

def get_rollout_context() -> Optional[RolloutContext]:
    return rollout_contextvar.get()

@dataclass
class GraderContext:
    rollout_id: str
    completion_callback_url: str
    samples: Dict[str, RolloutSample] = field(default_factory=dict)
    status: GraderStatus = GraderStatus.PENDING
    err_message: str | None = None
    err_category: RolloutErrorCategory | None = None

    def get_samples(self) -> Dict[str, RolloutSample]:
        '''
        Returns a dictionary of sample ids to samples

        Sample ids are usually the name/id of the agent, but if unspecified is a uuid string.
        '''
        return self.samples

    def set_sample_reward(self, sample_id: str, reward: float):
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        self.samples[sample_id].reward = reward
        return self.samples

    def set_grader_status(self, status: GraderStatus) -> None:
        self.status = status

    def set_grader_error(
        self, *, message: str, category: RolloutErrorCategory
    ) -> None:
        self.err_message = message
        self.err_category = category

    def make_grader_complete_request(self) -> GraderCompleteRequest:
        return GraderCompleteRequest(
            rollout_id=self.rollout_id,
            status=self.status,
            samples=self.samples,
            err_message=self.err_message,
            err_category=self.err_category,
        )

@dataclass
class AgentWorkflowContext(Generic[TConfig]):
    '''
    General context for an agent, agnostic of the rollout context.
    '''
    prompt: List[Dict[str, Any]]
    config: TConfig

    def __init__(self,
        prompt: List[Dict[str, Any]],
        config: TConfig,
    ):
        self.prompt = prompt
        self.config = config
