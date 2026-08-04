"""集中定义 v3 任务 kind — Plan §13；禁止魔法字符串散落。"""

IMAGE_GENERATION = "image.generation"
IMAGE_EDIT = "image.edit"
IMAGE_UPSCALE = "image.upscale"
VIDEO_GENERATION = "video.generation"
VIDEO_EDIT = "video.edit"
VIDEO_AVATAR = "video.avatar"
VIDEO_AVATAR_SCRIPT = "video.avatar_script"
VIDEO_UPSCALE = "video.upscale"
AUDIO_GENERATION = "audio.generation"
AUDIO_EDIT = "audio.edit"
LORA_TRAINING = "lora.training"
TOOLS_Z_IMAGE_MERGE = "tools.z_image_merge"

ALL_KINDS: frozenset[str] = frozenset(
    {
        IMAGE_GENERATION,
        IMAGE_EDIT,
        IMAGE_UPSCALE,
        VIDEO_GENERATION,
        VIDEO_EDIT,
        VIDEO_AVATAR,
        VIDEO_AVATAR_SCRIPT,
        VIDEO_UPSCALE,
        AUDIO_GENERATION,
        AUDIO_EDIT,
        LORA_TRAINING,
        TOOLS_Z_IMAGE_MERGE,
    }
)

# Registry ``actions`` keys → scheduler / API task ``kind`` strings.
REGISTRY_ACTION_TO_TASK_KIND: dict[str, dict[str, str]] = {
    "image": {
        "create": IMAGE_GENERATION,
        "rewrite": IMAGE_EDIT,
        "retouch": IMAGE_EDIT,
        "extend": IMAGE_EDIT,
        "upscale": IMAGE_UPSCALE,
    },
    "video": {
        "create": VIDEO_GENERATION,
        "animate": VIDEO_EDIT,
        "avatar": VIDEO_AVATAR,
        "avatar_script": VIDEO_AVATAR_SCRIPT,
        "upscale": VIDEO_UPSCALE,
    },
    "audio": {
        "create": AUDIO_GENERATION,
        "cover": AUDIO_EDIT,
        "repaint": AUDIO_EDIT,
    },
}


def task_kind_for_registry_action(media: str, action: str) -> str | None:
    """Map ``models_registry.json`` action name to ``task_kinds`` constant."""
    return REGISTRY_ACTION_TO_TASK_KIND.get(media, {}).get(action)


def is_image_kind(kind: str) -> bool:
    return kind.startswith("image.")


def is_video_kind(kind: str) -> bool:
    return kind.startswith("video.")


def is_audio_kind(kind: str) -> bool:
    return kind.startswith("audio.")


def is_lora_training_kind(kind: str) -> bool:
    return kind == LORA_TRAINING


def is_tools_kind(kind: str) -> bool:
    return kind.startswith("tools.")
