"""
持久化层实现
"""

import json
from typing import Dict, Any

from backend.core.interfaces import (
    IConfigStore, IPresetStore,
    AppSettings, IPathResolver
)
from backend.utils.workspace import is_workspace_configured

_SETTINGS_EXCLUDED_KEYS = frozenset(
    {
        "custom_workspace_dir",
        "custom_models_dir",
        "custom_loras_dir",
        "custom_outputs_dir",
    }
)


class JsonConfigStore(IConfigStore):
    """JSON配置存储"""
    
    def __init__(self, path_resolver: IPathResolver):
        self._path_resolver = path_resolver
        self._path = path_resolver.get_config_path()
    
    def _inject_workspace_dir(self, settings: AppSettings) -> AppSettings:
        """Workspace path comes from control-plane pointer only, not from ``.app_config.json``."""
        control = self._path_resolver.get_control_plane_dir()
        legacy = self._path_resolver.get_default_config_root()
        if is_workspace_configured(control, legacy_default_config=legacy):
            settings.custom_workspace_dir = str(self._path_resolver.get_project_root())
        else:
            settings.custom_workspace_dir = ""
        return settings

    def _settings_for_disk(self, settings: AppSettings) -> dict[str, Any]:
        return {
            k: v
            for k, v in settings.__dict__.items()
            if k in AppSettings.__dataclass_fields__ and k not in _SETTINGS_EXCLUDED_KEYS
        }

    def load(self) -> AppSettings:
        if not self._path.exists():
            return self._inject_workspace_dir(AppSettings())

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return self._inject_workspace_dir(AppSettings())
            allowed = {
                k: v
                for k, v in data.items()
                if k in AppSettings.__dataclass_fields__ and k not in _SETTINGS_EXCLUDED_KEYS
            }
            return self._inject_workspace_dir(AppSettings(**allowed))
        except Exception:
            return self._inject_workspace_dir(AppSettings())
    
    def save(self, settings: AppSettings) -> None:
        cfg_path = self._path_resolver.get_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self._settings_for_disk(settings), f, ensure_ascii=False, indent=2)
        self._path = cfg_path


class JsonPresetStore(IPresetStore):
    """JSON预设存储"""
    
    def __init__(self, path_resolver: IPathResolver):
        self._path = path_resolver.get_presets_path()
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        if not self._path.exists():
            return self._get_default_presets()
        
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._get_default_presets()
    
    def save(self, name: str, preset: Dict[str, Any]) -> None:
        presets = self.load_all()
        presets[name] = preset
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    
    def delete(self, name: str) -> None:
        presets = self.load_all()
        if name in presets:
            del presets[name]
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(presets, f, ensure_ascii=False, indent=2)
    
    def _get_default_presets(self) -> Dict[str, Dict[str, Any]]:
        preset_en = {
            "人像摄影": "Portrait Photography",
            "风景": "Landscape",
            "赛博朋克": "Cyberpunk",
            "写实增强": "Realism Enhancer",
            "动漫风格": "Anime Style",
            "油画艺术": "Oil Painting",
            "极简设计": "Minimalist Design",
            "产品摄影": "Product Photography",
            "食物摄影": "Food Photography",
            "室内设计": "Interior Design",
            "水彩画": "Watercolor",
            "黑白摄影": "B&W Photography",
            "浮世绘": "Ukiyo-e",
            "儿童插画": "Children's Illustration",
            "徽标设计": "Logo Design",
            "风格迁移-油画": "Style Transfer - Oil Painting",
            "风格迁移-水彩": "Style Transfer - Watercolor",
            "风格迁移-素描": "Style Transfer - Sketch",
            "风格迁移-3D卡通": "Style Transfer - 3D Cartoon",
            "细节增强": "Detail Enhancer",
            "背景替换-影棚": "BG Replace - Studio",
            "背景替换-自然": "BG Replace - Nature",
        }
        presets = {
            "人像摄影": {"applies_to": ["create"], "positive": "portrait of a person, soft natural lighting, detailed skin texture, shallow depth of field, bokeh background, 85mm lens, professional photography, 8k quality", "negative": "blurry, low quality, distorted face, extra limbs"},
            "风景": {"applies_to": ["create"], "positive": "a serene landscape, golden hour lighting, dramatic clouds, reflections in water, 8k quality, photorealistic, cinematic composition", "negative": "people, buildings, blurry, low quality"},
            "赛博朋克": {"applies_to": ["create"], "positive": "cyberpunk cityscape at night, neon lights, rain-soaked streets, reflections on wet pavement, flying cars, futuristic architecture, cinematic lighting, 8k", "negative": "daytime, sunny, blurry, low quality"},
            "写实增强": {"applies_to": ["create"], "positive": "photorealistic, detailed skin texture, natural lighting, professional photography, sharp focus, 8k quality, ultra detailed", "negative": "painting, drawing, anime, cartoon, blurry, low quality"},
            "动漫风格": {"applies_to": ["create"], "positive": "anime style, vibrant colors, detailed linework, soft cel shading, beautiful composition, high quality illustration, studio ghibli inspired", "negative": "realistic, photorealistic, 3d render, blurry, low quality, deformed"},
            "油画艺术": {"applies_to": ["create"], "positive": "oil painting, textured brushstrokes, rich colors, classical composition, impasto technique, masterpiece quality, canvas texture, gallery art", "negative": "photorealistic, digital art, 3d render, smooth, flat colors, low resolution"},
            "极简设计": {"applies_to": ["create"], "positive": "minimalist design, clean lines, geometric shapes, soft color palette, modern aesthetic, simple composition, elegant, high quality", "negative": "cluttered, busy, complex, messy, text, watermark"},
            "产品摄影": {"applies_to": ["create"], "positive": "commercial product photography, studio lighting, white background, sharp focus, professional, 8k, clean, premium quality", "negative": "blurry, messy background, text, watermark, low quality, distorted"},
            "食物摄影": {"applies_to": ["create"], "positive": "appetizing food photography, soft natural light, shallow depth of field, vibrant colors, gourmet plating, steam rising, rustic wooden table, 8k", "negative": "blurry, unappetizing, plastic, fake, text, watermark, low quality"},
            "室内设计": {"applies_to": ["create"], "positive": "interior design, modern architecture, natural light, elegant furniture, spacious room, high-end materials, architectural photography, 8k", "negative": "cluttered, dirty, dark, low ceiling, messy, people, low quality"},
            "水彩画": {"applies_to": ["create"], "positive": "watercolor painting, soft washes, transparent layers, delicate brushwork, artistic, dreamy atmosphere, paper texture, hand-painted", "negative": "digital art, sharp lines, photorealism, oil painting, thick paint, low quality"},
            "黑白摄影": {"applies_to": ["create"], "positive": "black and white photography, dramatic shadows, high contrast, fine art, timeless, grainy texture, film aesthetic, monochrome masterpiece", "negative": "color, low contrast, overexposed, blurry, digital artifacts"},
            "浮世绘": {"applies_to": ["create"], "positive": "ukiyo-e style, japanese woodblock print, flat colors, bold outlines, traditional composition, wave patterns, edo period aesthetic, artistic", "negative": "3d render, realistic, photorealistic, western painting, perspective depth, low quality"},
            "儿童插画": {"applies_to": ["create"], "positive": "children's book illustration, whimsical, cute characters, colorful, storybook style, soft shapes, playful atmosphere, hand-drawn charm", "negative": "scary, dark, realistic, adult content, blurry, low quality"},
            "徽标设计": {"applies_to": ["create"], "positive": "logo design, minimalist, vector style, clean lines, professional branding, flat colors, balanced composition, modern icon", "negative": "photorealistic, complex, text, watermark, cluttered, gradients, 3d"},
            "风格迁移-油画": {"applies_to": ["rewrite"], "positive": "oil painting style, textured brushstrokes, rich impasto, classical technique, artistic interpretation, canvas texture", "negative": "photorealistic, digital art, sharp details, smooth surface, low quality"},
            "风格迁移-水彩": {"applies_to": ["rewrite"], "positive": "watercolor painting style, soft washes, transparent layers, delicate brushwork, paper texture, artistic flow", "negative": "photorealistic, sharp lines, digital art, thick paint, oil painting, low quality"},
            "风格迁移-素描": {"applies_to": ["rewrite"], "positive": "pencil sketch, detailed linework, cross-hatching, monochrome, artistic drawing, graphite texture, hand-drawn style", "negative": "color, photorealistic, painting, smooth, digital rendering, blurry"},
            "风格迁移-3D卡通": {"applies_to": ["rewrite"], "positive": "pixar style, 3d cartoon render, smooth surfaces, vibrant colors, cute proportions, soft lighting, animated film quality, playful", "negative": "realistic, photorealistic, dark, scary, low quality, distorted"},
            "细节增强": {"applies_to": ["rewrite"], "positive": "enhanced detail, sharp focus, 8k upscale, refined texture, high definition, professional quality, improved clarity, ultra detailed", "negative": "blurry, pixelated, noise, artifacts, over-sharpened, halo effects, low quality"},
            "背景替换-影棚": {"applies_to": ["rewrite"], "positive": "studio lighting, professional backdrop, clean background, soft shadows, commercial photography, well-lit, high key", "negative": "messy background, cluttered, outdoor, dark, blurry, low quality"},
            "背景替换-自然": {"applies_to": ["rewrite"], "positive": "natural outdoor background, golden hour lighting, bokeh effect, lush greenery, shallow depth of field, scenic backdrop", "negative": "indoor, studio, artificial lighting, urban, messy, low quality"},
        }
        for key, val in presets.items():
            val["name_en"] = preset_en.get(key, key)
            app = val.get("applies_to")
            if isinstance(app, list) and "animate" in app:
                val["media_scope"] = "video"
            else:
                val["media_scope"] = "image"
        return presets
