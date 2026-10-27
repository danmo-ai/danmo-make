"""ILoraTrainEngine — LoRA training + checkpoint candidate evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from backend.core.contracts import (
    EngineResult,
    ExecutionContext,
    LoraEvaluationRequest,
    LoraTrainingRequest,
)


class ILoraTrainEngine(ABC):
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def supports_base_model(self, base_model_id: str) -> bool:
        pass

    @abstractmethod
    async def train(self, request: LoraTrainingRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def evaluate(self, request: LoraEvaluationRequest, ctx: ExecutionContext) -> EngineResult:
        pass
