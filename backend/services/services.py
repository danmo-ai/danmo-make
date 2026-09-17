"""
Service layer implementation — settings (generation tasks handled by TaskScheduler + media routing)
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.core.interfaces import (
    ISettingsService,
    IConfigStore,
    IPathResolver,
    AppSettings,
    ModelInfo,
    ModelConfig,
)
from backend.core.media_interfaces import IImageEngine, IVideoEngine
from backend.utils.path_utils import get_system_info
from backend.core.registry_format import (
    media_from_record,
    parse_i18n_string_pair,
    typed_parameters,
)


class SettingsService(ISettingsService):
    """Settings service implementation"""
    
    def __init__(
        self,
        config_store: IConfigStore,
        path_resolver: IPathResolver,
        image_engine: IImageEngine,
        video_engine: IVideoEngine,
    ):
        self._config = config_store
        self._path_resolver = path_resolver
        self._image_engine = image_engine
        self._video_engine = video_engine
    
    def get_settings(self) -> AppSettings:
        return self._config.load()
    
    def update_settings(self, settings: AppSettings) -> None:
        self._config.save(settings)
    
    def get_available_models(self) -> List[ModelInfo]:
        models = []
        models_dir = self._path_resolver.get_models_dir()
        registry = self.get_model_registry()
        
        if not models_dir.exists():
            return models
        
        category_type_map = {
            'base': 'diffusion', 'base_models': 'diffusion',
            'controlnet': 'controlnet', 'controlnets': 'controlnet',
            'upscaler': 'upscaler', 'upscalers': 'upscaler',
            'tool': 'tool', 'tools': 'tool',
            'lora': 'lora', 'loras': 'lora',
        }
        
        valid_extensions = [".safetensors", ".bin", ".pt"]
        model_indicators = ["model_index.json"]
        
        def is_model_file(f: Path) -> bool:
            return f.suffix in valid_extensions or f.name in model_indicators
        
        for subdir in models_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            
            category_name = subdir.name.lower()
            category_type = category_type_map.get(category_name, 'unknown')
            
            # 1. Scan subdirectories (third-level directories as model packages)
            for model_dir in subdir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith('.'):
                    continue
                has_files = any(is_model_file(f) for f in model_dir.rglob("*"))
                if has_files:
                    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    model_type = category_type
                    if category_name in ['base', 'base_models'] and model_dir.name in registry:
                        model_type = registry[model_dir.name].type or category_type
                    models.append(ModelInfo(name=model_dir.name, path=str(model_dir), type=model_type, size=size))
            
            # 2. Scan direct files (third-level files as single-file models)
            for f in subdir.iterdir():
                if f.is_file() and not f.name.startswith('.') and f.suffix in valid_extensions:
                    models.append(ModelInfo(
                        name=f.stem,
                        path=str(f),
                        type=category_type,
                        size=f.stat().st_size
                    ))
        
        return models
    
    def refresh_models(self) -> None:
        # Rescan
        pass
    
    def install_environment(self) -> bool:
        # TODO: implement environment installation
        return False
    
    def check_environment(self) -> bool:
        return self._image_engine.is_available() or self._video_engine.is_available()
    
    def get_system_info(self) -> dict:
        return get_system_info()

    def get_model_registry(self) -> Dict[str, ModelConfig]:
        """Read models_registry.json to get all model configurations"""
        registry_path = self._path_resolver.get_models_registry_path()
        if not registry_path.exists():
            return {}
        try:
            from backend.catalog.loader import expand_catalog_document

            with open(registry_path, "r", encoding="utf-8") as f:
                data = expand_catalog_document(json.load(f))
            models = {}
            for key, val in data.get("models", {}).items():
                if not isinstance(val, dict):
                    continue
                nz, ne, dz, de = parse_i18n_string_pair(
                    val.get("name"),
                    val.get("description"),
                    key,
                )
                media = media_from_record(val)
                params = typed_parameters(val.get("parameters") or {})
                acts = val.get("actions") if isinstance(val.get("actions"), dict) else {}
                neg = val.get("negative_prompt_support")
                if not isinstance(neg, bool):
                    p0 = val.get("parameters") or {}
                    neg = bool(p0.get("negative_prompt_support")) if isinstance(p0, dict) else False
                cup_raw = val.get("commercial_use_allowed")
                if cup_raw is True:
                    commercial_use_allowed = True
                elif cup_raw is False:
                    commercial_use_allowed = False
                else:
                    commercial_use_allowed = None
                models[key] = ModelConfig(
                    name={"zh": nz, "en": ne},
                    description={"zh": dz, "en": de},
                    engine=val.get("engine", ""),
                    type=val.get("type", ""),
                    family=str(val.get("family") or ""),
                    parameters=params,
                    recommended=val.get("recommended", False),
                    category=val.get("category", "base_models"),
                    dependencies=val.get("dependencies", []),
                    source=val.get("source", "huggingface"),
                    download_url=val.get("download_url"),
                    files=val.get("files"),
                    versions=val.get("versions"),
                    negative_prompt_support=neg,
                    base_model=val.get("base_model"),
                    nsfw=val.get("nsfw", False),
                    commercial_use_allowed=commercial_use_allowed,
                    media=media,
                    actions=dict(acts),
                    stub_no_download=bool(val.get("stub_no_download")),
                )
            return models
        except Exception as e:
            print(f"Failed to read model registry: {e}")
            return {}

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get single model configuration"""
        registry = self.get_model_registry()
        return registry.get(model_name)

    @staticmethod
    def _path_has_bundle_weights(model_dir: Path) -> bool:
        if model_dir.is_file():
            return bool(
                model_dir.suffix in [".safetensors", ".bin", ".pt", ".ckpt"]
                or model_dir.name
                in ["model_index.json", "config.json", "pytorch_model.bin.index.json"]
            )
        return any(
            f.suffix in [".safetensors", ".bin", ".pt", ".ckpt"]
            or f.name in ["model_index.json", "config.json", "pytorch_model.bin.index.json"]
            for f in model_dir.rglob("*")
            if f.is_file()
        )

    def _resolve_registry_version_bundle_dir(
        self,
        model_name: str,
        version_key: str,
        version_config: Any,
    ) -> Path:
        """Resolve bundle path: ``bundle_repos[0].local_path``, ``local_path``, or fallback id."""
        from backend.core.bundle_repos import version_primary_local_path

        vc = version_config if isinstance(version_config, dict) else {}
        try:
            local_path = version_primary_local_path(vc)
        except ValueError:
            local_path = f"models/{model_name}-{version_key}"
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _registry_has_any_ready_bundle(self, model_name: str, config: ModelConfig) -> bool:
        """True when registry ``versions`` point at existing paths with weight files (same rule as detailed status)."""
        if getattr(config, "stub_no_download", False):
            return True
        versions = getattr(config, "versions", None) or {}
        from backend.core.bundle_repos import bundle_local_paths, version_primary_local_path

        for version_key, version_config in versions.items():
            vc = version_config if isinstance(version_config, dict) else {}
            try:
                repo_paths = bundle_local_paths(vc)
                if not repo_paths:
                    repo_paths = [version_primary_local_path(vc)]
            except ValueError:
                repo_paths = [f"models/{model_name}-{version_key}"]

            all_ready = True
            for local_path in repo_paths:
                model_dir = self._path_resolver.resolve_registry_local_path(local_path)
                if not model_dir.exists() or not self._path_has_bundle_weights(model_dir):
                    all_ready = False
                    break
            if all_ready:
                return True
        return False

    def lora_adapter_picklist(self, for_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Adapter picklist of installed LoRAs; filters by base_model when ``for_model`` is an image model id."""
        registry = self.get_model_registry()
        out: List[Dict[str, Any]] = []
        for key, config in registry.items():
            if getattr(config, "category", None) != "loras":
                continue
            if not self._registry_has_any_ready_bundle(key, config):
                continue
            lora_base = config.base_model or ""
            if for_model:
                model_base_key = for_model.split(":", 1)[0].strip()
                lora_base_key = lora_base.split(":", 1)[0].strip() if lora_base else ""
                if model_base_key.startswith("flux2"):
                    ok = (
                        lora_base_key == ""
                        or lora_base_key.startswith("flux2")
                        or lora_base_key == model_base_key
                    )
                elif model_base_key.startswith("flux1"):
                    ok = (
                        lora_base_key == ""
                        or lora_base_key.startswith("flux1")
                        or lora_base_key == model_base_key
                    )
                elif model_base_key.startswith("flux"):
                    ok = (
                        lora_base_key == ""
                        or lora_base_key.startswith("flux")
                        or lora_base_key == model_base_key
                    )
                elif model_base_key.startswith("wan"):
                    from backend.engine.families.wan.lora_mlx import wan_video_lora_base_compatible

                    ok = lora_base_key == "" or wan_video_lora_base_compatible(
                        model_base_key, lora_base_key
                    )
                elif model_base_key.startswith("qwen") or model_base_key.startswith("firered-image-edit"):
                    from backend.engine.families.qwen.weights_mlx import qwen_image_lora_base_compatible

                    ok = lora_base_key == "" or qwen_image_lora_base_compatible(
                        model_base_key, lora_base_key
                    )
                elif model_base_key.startswith("ace-step"):
                    from backend.engine.families.ace_step.weights import ace_step_lora_base_compatible

                    ok = lora_base_key == "" or ace_step_lora_base_compatible(
                        model_base_key, lora_base_key
                    )
                else:
                    from backend.engine.families.z_image.weights import z_image_lora_base_compatible

                    ok = lora_base_key == "" or z_image_lora_base_compatible(
                        model_base_key, lora_base_key
                    )
                if not ok:
                    continue
            from backend.catalog.lora_meta import lora_adapter_picklist_row, lora_config_picklist_extras

            out.append(
                lora_adapter_picklist_row(
                    config,
                    lora_id=key,
                    name=(config.name or {}).get("zh") or (config.name or {}).get("en") or key,
                    base_model=lora_base,
                    source="registry",
                )
            )
        from backend.engine.training.user_lora_registry import list_user_loras

        config_dir = self._path_resolver.get_workspace_config_dir()
        for ul in list_user_loras(config_dir):
            lora_base = str(ul.get("base_model") or "")
            if for_model:
                model_base_key = for_model.split(":", 1)[0].strip()
                lora_base_key = lora_base.split(":", 1)[0].strip() if lora_base else ""
                if lora_base_key and lora_base_key != model_base_key:
                    from backend.engine.families.z_image.weights import z_image_lora_base_compatible

                    if model_base_key.startswith("flux1") and lora_base_key.startswith("flux1"):
                        pass
                    elif model_base_key.startswith("qwen") or model_base_key.startswith("firered-image-edit"):
                        from backend.engine.families.qwen.weights_mlx import qwen_image_lora_base_compatible

                        if qwen_image_lora_base_compatible(model_base_key, lora_base_key):
                            pass
                        else:
                            continue
                    elif model_base_key.startswith("ace-step"):
                        from backend.engine.families.ace_step.weights import ace_step_lora_base_compatible

                        if ace_step_lora_base_compatible(model_base_key, lora_base_key):
                            pass
                        else:
                            continue
                    elif model_base_key.startswith("wan"):
                        from backend.engine.families.wan.lora_mlx import wan_video_lora_base_compatible

                        if wan_video_lora_base_compatible(model_base_key, lora_base_key):
                            pass
                        else:
                            continue
                    elif z_image_lora_base_compatible(model_base_key, lora_base_key):
                        pass
                    else:
                        continue
            local_path = str(ul.get("local_path") or "")
            if local_path:
                resolved = self._path_resolver.resolve_registry_local_path(local_path)
                if not resolved.is_dir() and not resolved.is_file():
                    continue
            row: dict[str, Any] = {
                "kind": "lora",
                "id": str(ul.get("id")),
                "name": str(ul.get("name") or ul.get("id")),
                "base_model": lora_base,
                "source": str(ul.get("source") or "user_trained"),
                "local_path": local_path,
            }
            if local_path:
                bundle_dir = resolved if resolved.is_dir() else resolved.parent
                row.update(lora_config_picklist_extras(bundle_dir))
            if "hint_key" not in row and lora_base.split(":", 1)[0].strip() == "z-image-turbo":
                from backend.catalog.lora_meta import z_image_turbo_trained_picklist_extras

                row.update(z_image_turbo_trained_picklist_extras())
            out.insert(
                0,
                row,
            )
        return out

    def get_models_status(self) -> Dict[str, bool]:
        """Get readiness status for all registered models.
        
        Returns:
            model_name -> readiness dict
        """
        # Reuse detailed status logic to avoid inconsistent path resolution
        detailed = self.get_models_detailed_status()
        return {k: v['ready'] for k, v in detailed.items()}
    
    def get_models_detailed_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed model status (version-level check, distinguishes not_downloaded/incomplete/ready)"""
        status = {}
        registry = self.get_model_registry()
        for model_name, config in registry.items():
            if getattr(config, "stub_no_download", False):
                status[model_name] = {
                    "versions": {
                        "stub": {
                            "status": "ready",
                            "label": "Stub (no download)",
                            "ready": True,
                        }
                    },
                    "ready": True,
                    "status": "ready",
                }
                continue
            versions = getattr(config, 'versions', None) or {}
            model_status = {
                "versions": {},
                "ready": False,
                "status": "not_downloaded"
            }
            any_ready = False
            
            for version_key, version_config in versions.items():
                vc = version_config if isinstance(version_config, dict) else {}
                from backend.core.bundle_repos import bundle_local_paths, version_primary_local_path

                try:
                    repo_paths = bundle_local_paths(vc)
                    if not repo_paths:
                        repo_paths = [version_primary_local_path(vc)]
                except ValueError:
                    repo_paths = [f"models/{model_name}-{version_key}"]

                missing_paths: list[str] = []
                for local_path in repo_paths:
                    model_dir = self._path_resolver.resolve_registry_local_path(local_path)
                    if not model_dir.exists() or not self._path_has_bundle_weights(model_dir):
                        missing_paths.append(local_path)

                if missing_paths:
                    model_status["versions"][version_key] = {
                        "status": "not_downloaded" if all(
                            not self._path_resolver.resolve_registry_local_path(p).exists()
                            for p in repo_paths
                        ) else "incomplete",
                        "label": "Not downloaded" if len(missing_paths) == len(repo_paths) else "Files missing",
                        "ready": False,
                        "missing_paths": missing_paths,
                    }
                    continue

                model_dir = self._resolve_registry_version_bundle_dir(
                    model_name, version_key, version_config
                )
                has_weights = self._path_has_bundle_weights(model_dir)
                family = config.family or ""
                category = getattr(config, "category", None) or ""
                components: dict[str, Any] | None = None
                from backend.core.bundle_manifest import (
                    bundle_component_status,
                    skips_full_family_bundle_contract,
                )

                if has_weights and family and not skips_full_family_bundle_contract(category):
                    components = bundle_component_status(
                        model_dir,
                        family=family,
                        version_config=vc,
                        project_root=self._path_resolver.get_project_root(),
                    )

                version_entry: dict[str, Any] = {
                    "status": "not_downloaded",
                    "label": "Not downloaded",
                    "ready": False,
                }
                split_bundle = len(repo_paths) > 1
                if split_bundle:
                    # bundle_repos may split DiT/VAE/SigLIP from external text encoders
                    # (e.g. Hunyuan I2V Qwen + ByT5). All paths were validated above.
                    version_entry = {
                        "status": "ready",
                        "label": "Ready",
                        "ready": True,
                    }
                    if components is not None:
                        version_entry["bundle_components"] = components
                    any_ready = True
                elif has_weights:
                    if components is not None and not components.get("complete", True):
                        version_entry = {
                            "status": "incomplete",
                            "label": "Missing components",
                            "ready": False,
                            "bundle_components": components,
                        }
                    else:
                        version_entry = {
                            "status": "ready",
                            "label": "Ready",
                            "ready": True,
                        }
                        if components is not None:
                            version_entry["bundle_components"] = components
                        any_ready = True
                else:
                    version_entry = {
                        "status": "incomplete",
                        "label": "Files missing",
                        "ready": False,
                    }
                model_status["versions"][version_key] = version_entry
            
            # Set overall model status
            if any_ready:
                model_status["ready"] = True
                model_status["status"] = "ready"
            elif model_status["versions"]:
                # Check if all versions are not downloaded
                all_not_downloaded = all(v["status"] == "not_downloaded" for v in model_status["versions"].values())
                if all_not_downloaded:
                    model_status["status"] = "not_downloaded"
                else:
                    model_status["status"] = "incomplete"
            
            status[model_name] = model_status
        
        return status
    
    def get_disk_space(self) -> Dict[str, Any]:
        """Get disk space info"""
        import shutil
        
        # Get directory paths
        models_dir = self._path_resolver.get_models_dir()
        loras_dir = self._path_resolver.get_loras_dir()
        outputs_dir = self._path_resolver.get_outputs_dir()
        
        # Get disk usage
        def get_dir_info(path: Path) -> Dict[str, Any]:
            if not path.exists():
                return {"exists": False, "size": 0, "size_human": "0 B"}
            
            # Calculate directory size
            total_size = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
            
            # Get disk free space
            usage = shutil.disk_usage(str(path))
            
            return {
                "exists": True,
                "size": total_size,
                "size_human": _human_readable_size(total_size),
                "free": usage.free,
                "free_human": _human_readable_size(usage.free),
                "total": usage.total,
                "total_human": _human_readable_size(usage.total),
                "percent_used": round((usage.total - usage.free) / usage.total * 100, 1)
            }
        
        return {
            "models": get_dir_info(models_dir),
            "loras": get_dir_info(loras_dir),
            "outputs": get_dir_info(outputs_dir),
        }


def _human_readable_size(size_bytes: int) -> str:
    """Convert to human-readable size"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

