"""
v3 media engine interfaces: 1:1 aligned with REST endpoints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    EngineResult,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
    VideoAvatarRequest,
    VideoAvatarScriptRequest,
    VideoEditRequest,
    VideoGenerationRequest,
    VideoUpscaleRequest,
)


class IImageEngine(ABC):
    media_type: ClassVar[str] = "image"
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass

    @abstractmethod
    def supports(self, model_id: str, action: str) -> bool:
        """action: generate | edit | upscale"""
        ...

    @abstractmethod
    async def generate(
        self, request: ImageGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        pass

    @abstractmethod
    async def edit(self, request: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def upscale(self, request: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass


class IVideoEngine(ABC):
    media_type: ClassVar[str] = "video"
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass

    @abstractmethod
    def supports(self, model_id: str, action: str) -> bool:
        """action: generate | edit | avatar | avatar_script | upscale"""
        ...

    @abstractmethod
    async def generate(
        self, request: VideoGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        pass

    @abstractmethod
    async def edit(self, request: VideoEditRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def avatar(self, request: VideoAvatarRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def avatar_script(
        self, request: VideoAvatarScriptRequest, ctx: ExecutionContext
    ) -> EngineResult:
        """Text-script-driven avatar video (TTS → lip-sync). Phase 1 placeholder."""
        pass

    @abstractmethod
    async def upscale(self, request: VideoUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass


class IAudioEngine(ABC):
    media_type: ClassVar[str] = "audio"
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass

    @abstractmethod
    def supports(self, model_id: str, action: str) -> bool:
        """action: create_music | edit"""
        ...

    @abstractmethod
    async def generate(
        self, request: AudioGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        pass

    @abstractmethod
    async def edit(self, request: AudioEditRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass
