"""Smoke tests for engine sessions, protocols, platform, and inference."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.engine.inference import DiffusionInference, ParadigmBundle
from backend.engine.platform.fake import fake_platform
from backend.engine.platform.session import platform_from_runtime
from backend.engine.protocols import FamilySpec
from backend.engine.registry.bootstrap import bootstrap_family_plugins
from backend.engine.registry.family_registry import (
    build_family_plugin,
    is_family_plugin_registered,
    register_family,
)
from backend.engine.sessions import (
    routes_to_audio_edit_session,
    routes_to_audio_session,
    routes_to_image_edit_session,
    routes_to_image_session,
    routes_to_upscale_session,
    routes_to_video_session,
)
from backend.engine.sessions._phases.resolve import resolve_phase


class EngineSessionsTests(unittest.TestCase):
    def test_family_spec_defaults(self) -> None:
        spec = FamilySpec(family_id="flux2", media="image")
        self.assertEqual(spec.paradigm, "diffusion")
        self.assertEqual(spec.cfg_mode, "dual")

    def test_family_spec_from_catalog_flux2(self) -> None:
        from backend.catalog.family_spec_loader import family_spec_from_catalog
        from backend.engine.config.model_configs import get_config_class

        config = get_config_class("flux2")()
        spec = family_spec_from_catalog("flux2", config=config, media="image")
        self.assertEqual(spec.paradigm, "diffusion")
        self.assertEqual(spec.step_kwargs_profile, "flux2")
        self.assertIn("lora_merge", spec.hooks)

    def test_family_spec_from_catalog_ltx_two_stage(self) -> None:
        from backend.catalog.family_spec_loader import family_spec_from_catalog
        from backend.engine.config.model_configs import get_config_class

        config = get_config_class("ltx")()
        spec = family_spec_from_catalog("ltx", config=config, media="video")
        self.assertEqual(spec.paradigm, "two_stage")
        self.assertEqual(spec.media, "video")

    def test_all_plugin_families_in_catalog(self) -> None:
        from backend.catalog.family_spec_loader import load_families_block
        from backend.engine.config.model_configs import FAMILY_CONFIG_MAP
        from backend.engine.registry.bootstrap import bootstrap_family_plugins
        from backend.engine.registry.family_registry import registered_family_ids

        bootstrap_family_plugins()
        families = load_families_block()
        for family_id in FAMILY_CONFIG_MAP:
            self.assertIn(
                family_id,
                families,
                msg=f"catalog families.{family_id} missing",
            )
        for family_id in registered_family_ids():
            self.assertIn(family_id, families)

    def test_fake_platform_kernels(self) -> None:
        plat = fake_platform()
        t = plat.kernels.randn((2, 3))
        self.assertEqual(tuple(t.shape), (2, 3))

    def test_platform_from_runtime(self) -> None:
        runtime = MagicMock()
        runtime.backend = "mlx"
        plat = platform_from_runtime(runtime)
        self.assertTrue(plat.is_mlx)
        self.assertIs(plat.kernels, runtime)

    def test_diffusion_inference_construct(self) -> None:
        kernels = MagicMock()
        infer = DiffusionInference(kernels)
        self.assertIs(infer._ctx, kernels)

    def test_run_job_bundle(self) -> None:
        from backend.engine.inference.job import JobBundle, run_job

        sentinel = object()
        run_fn = MagicMock(return_value=sentinel)
        out = run_job(JobBundle(run_fn=run_fn, kwargs={"scale": 2}))
        self.assertIs(out, sentinel)
        run_fn.assert_called_once_with(scale=2)

    def test_paradigm_bundle_alias(self) -> None:
        self.assertIs(ParadigmBundle, __import__(
            "backend.engine.inference._protocols", fromlist=["InferenceBundle"]
        ).InferenceBundle)

    def test_family_registry_fail_loud(self) -> None:
        with self.assertRaises(RuntimeError):
            build_family_plugin(
                "nonexistent_family",
                fake_platform(),
                model_id="x",
                bundle_root=Path("/tmp"),
            )

    def test_resolve_phase_without_plugin(self) -> None:
        runtime = MagicMock()
        runtime.backend = "mlx"
        session = MagicMock()
        session._project_root = Path.cwd()
        session._registry.require.return_value = MagicMock(
            family="flux2",
            raw={},
        )
        request = MagicMock()
        request.model = "flux2-dev"
        exec_ctx = MagicMock()
        with unittest.mock.patch(
            "backend.engine.sessions._phases.resolve.local_bundle_root",
            return_value=Path("/tmp/bundle"),
        ):
            with unittest.mock.patch(
                "backend.engine.sessions._phases.resolve.require_entry_family",
                return_value="flux2",
            ):
                resolved = resolve_phase(session, request, exec_ctx, runtime_ctx=runtime)
        self.assertEqual(resolved.family_id, "flux2")
        self.assertIsNone(resolved.plugin)

    def test_flux2_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("flux2"))
        plugin = build_family_plugin(
            "flux2",
            fake_platform(),
            model_id="flux2-dev",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.family_id, "flux2")
        self.assertEqual(plugin.spec.vae_scale, 16)

    def test_qwen_image_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("qwen_image"))
        plugin = build_family_plugin(
            "qwen_image",
            fake_platform(),
            model_id="qwen-image",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.family_id, "qwen_image")
        self.assertEqual(plugin.spec.latent_layout, "qwen_grid")
        self.assertEqual(plugin.spec.vae_scale, 16)

    def test_flux1_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("flux1"))
        plugin = build_family_plugin(
            "flux1",
            fake_platform(),
            model_id="flux1-schnell",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.family_id, "flux1")
        self.assertEqual(plugin.spec.latent_layout, "packed_seq")

    def test_z_image_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("z_image"))
        plugin = build_family_plugin(
            "z_image",
            fake_platform(),
            model_id="z-image-turbo",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.family_id, "z_image")
        self.assertEqual(plugin.spec.vae_scale, 8)
        self.assertEqual(plugin.spec.latent_layout, "nchw")

    def test_routes_to_image_session(self) -> None:
        bootstrap_family_plugins()
        registry = MagicMock()
        registry.get.return_value = MagicMock(
            family="flux2", media="image", actions=frozenset({"generate"})
        )
        self.assertTrue(routes_to_image_session("flux2-dev", registry))
        registry.get.return_value = MagicMock(
            family="z_image", media="image", actions=frozenset({"generate"})
        )
        self.assertTrue(routes_to_image_session("z-image-turbo", registry))
        registry.get.return_value = MagicMock(
            family="wan", media="video", actions=frozenset({"generate"})
        )
        self.assertFalse(routes_to_image_session("wan-2.2", registry))

    def test_wan_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("wan"))
        plugin = build_family_plugin(
            "wan",
            fake_platform(),
            model_id="wan-2.2-ti2v-5b",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.media, "video")
        self.assertEqual(plugin.spec.paradigm, "diffusion")

    def test_ltx_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("ltx"))
        plugin = build_family_plugin(
            "ltx",
            fake_platform(),
            model_id="ltx-2.3-dev",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.media, "video")
        self.assertEqual(plugin.spec.paradigm, "two_stage")
        self.assertEqual(plugin.spec.vae_scale, 32)

    def test_ace_step_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("ace_step"))
        plugin = build_family_plugin(
            "ace_step",
            fake_platform(),
            model_id="ace-step-xl-sft",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.media, "audio")
        self.assertEqual(plugin.spec.paradigm, "flow_matching")
        self.assertEqual(plugin.spec.latent_layout, "audio_1d")

    def test_diffrhythm_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("diffrhythm"))
        plugin = build_family_plugin(
            "diffrhythm",
            fake_platform(),
            model_id="diffrhythm-v2",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.media, "audio")
        self.assertEqual(plugin.spec.paradigm, "block_ar")
        self.assertEqual(plugin.spec.latent_layout, "audio_1d")

    def test_hunyuan_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("hunyuan"))
        plugin = build_family_plugin(
            "hunyuan",
            fake_platform(),
            model_id="hunyuan-video-1.5",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.media, "video")
        self.assertEqual(plugin.spec.paradigm, "diffusion")
        self.assertEqual(plugin.spec.vae_scale, 16)

    def test_seedvr2_plugin_bootstrap(self) -> None:
        bootstrap_family_plugins()
        self.assertTrue(is_family_plugin_registered("seedvr2"))
        plugin = build_family_plugin(
            "seedvr2",
            fake_platform(),
            model_id="seedvr2-3b",
            bundle_root=Path("/tmp/bundle"),
        )
        self.assertEqual(plugin.spec.paradigm, "job")

    def test_video_and_upscale_session_routing(self) -> None:
        bootstrap_family_plugins()
        registry = MagicMock()
        registry.get.return_value = MagicMock(family="wan", media="video")
        self.assertTrue(routes_to_video_session("wan-2.2-ti2v-5b", registry))
        registry.get.return_value = MagicMock(family="ltx", media="video")
        self.assertTrue(routes_to_video_session("ltx-2.3-dev", registry))
        registry.get.return_value = MagicMock(family="hunyuan", media="video")
        self.assertTrue(routes_to_video_session("hunyuan-video-1.5", registry))
        registry.get.return_value = MagicMock(
            family="ace_step",
            media="audio",
            actions=frozenset({"create_music", "edit"}),
        )
        self.assertTrue(routes_to_audio_session("ace-step-xl-sft", registry))
        self.assertTrue(routes_to_audio_edit_session("ace-step-xl-sft", registry))
        registry.get.return_value = MagicMock(
            family="flux2", media="image", actions=frozenset({"edit"})
        )
        self.assertTrue(routes_to_image_edit_session("flux2-dev", registry))
        registry.get.return_value = MagicMock(
            family="diffrhythm",
            media="audio",
            actions=frozenset({"create_music"}),
        )
        self.assertTrue(routes_to_audio_session("diffrhythm-v2", registry))
        registry.get.return_value = MagicMock(
            family="seedvr2", media="image", actions=frozenset({"upscale"})
        )
        self.assertTrue(routes_to_upscale_session("seedvr2-3b", registry))
        self.assertFalse(routes_to_image_session("seedvr2-3b", registry))
        from backend.engine.sessions.video_upscale_session import routes_to_video_upscale_session

        registry.get.return_value = MagicMock(
            id="hunyuan-video-1.5-1080p-sr",
            family="hunyuan",
            media="video",
            raw={"versions": {"default": {"hunyuan_ms_variant": "1080p_sr_distilled"}}},
        )
        self.assertTrue(routes_to_video_upscale_session("hunyuan-video-1.5-1080p-sr", registry))
        registry.get.return_value = MagicMock(
            id="seedvr2-video-7b",
            family="seedvr2",
            media="video",
            raw={"versions": {"fp16": {"video_upscale_kind": "seedvr2_spatiotemporal"}}},
        )
        self.assertTrue(routes_to_video_upscale_session("seedvr2-video-7b:fp16", registry))

    def test_schedule_phase_from_create_ctx(self) -> None:
        from backend.engine.pipelines.image_create_phases import ImageCreateRunContext
        from backend.engine.sessions._phases.schedule import schedule_phase

        semantics = type("S", (), {"scheduler_name": "flow_match_euler"})()
        ctx = ImageCreateRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            config=MagicMock(),
            runtime_contract=MagicMock(),
            family="flux2",
            model_key="flux2-klein-9b",
            version_key=None,
            bundle_root=None,
            model=MagicMock(),
            extra_cond={},
            txt_embeds=None,
            neg_embeds=None,
            txt_attn_mask=None,
            neg_attn_mask=None,
            pooled_embeds=None,
            neg_pooled_embeds=None,
            encoder_type="flux2",
            scheduler=MagicMock(),
            timesteps=[1.0, 0.5],
            sigmas=None,
            sched_ts=None,
            timestep_embed_schedule=None,
            semantics=semantics,
            w=512,
            h=512,
            steps=4,
            guidance=1.0,
            base_seed=42,
            n=1,
            preview_mode="none",
            preview_interval=2,
            preview_max_edge=512,
            preview_decoder="none",
            preview_state={},
            latent_noise_dtype=None,
            noise_sample_dtype=None,
            packed_denoise=False,
            flux_pack=None,
            flux_unpack=None,
            latent_h=0,
            latent_w=0,
            packed_shape=None,
            structural_output_meta=None,
        )
        resolved = MagicMock()
        state = schedule_phase(resolved, ctx=ctx)
        self.assertEqual(state.scheduler_name, "flow_match_euler")
        self.assertEqual(state.timesteps, [1.0, 0.5])

    def test_register_family_builder(self) -> None:
        def _builder(platform, *, model_id, bundle_root, version_key=None):
            from backend.engine.protocols import FamilyPlugin

            spec = FamilySpec(family_id="test_sessions", media="image")
            backbone = MagicMock()
            return FamilyPlugin(family_id="test_sessions", spec=spec, backbone=backbone)

        register_family("test_sessions", _builder)
        plugin = build_family_plugin(
            "test_sessions",
            fake_platform(),
            model_id="m",
            bundle_root=Path("/tmp"),
        )
        self.assertEqual(plugin.family_id, "test_sessions")

    def test_video_plugin_skips_backbone_load_for_ltx(self) -> None:
        from backend.engine.families._video_backbone import VideoPluginBackbone
        from backend.engine.protocols import FamilyPlugin, FamilySpec

        spec = FamilySpec(family_id="ltx", media="video")
        backbone = VideoPluginBackbone(spec)
        entry = MagicMock()
        entry.parameters = {}
        entry.raw = {}
        request = MagicMock(num_frames=None)
        should_load = backbone.bind_load_context(
            registry_entry=entry,
            project_root=Path("/tmp"),
            model_cache=None,
            bundle_root=Path("/tmp/bundle"),
            request=request,
        )
        self.assertFalse(should_load)
        self.assertTrue(backbone._skip_load)

    def test_image_plugin_skips_backbone_load_for_hidream_o1(self) -> None:
        from backend.engine.families._image_backbone import ImagePluginBackbone
        from backend.engine.protocols import FamilySpec

        spec = FamilySpec(family_id="hidream_o1", media="image")
        backbone = ImagePluginBackbone(spec)
        entry = MagicMock()
        entry.parameters = {}
        entry.raw = {}
        should_load = backbone.bind_load_context(
            registry_entry=entry,
            project_root=Path("/tmp"),
            model_cache=None,
            bundle_root=Path("/tmp/bundle"),
            request=MagicMock(),
        )
        self.assertFalse(should_load)
        self.assertTrue(backbone._skip_load)

    def test_assert_generation_family_missing_plugin_fails_loud(self) -> None:
        from unittest.mock import patch

        from backend.engine.sessions.engine_dispatch import assert_generation_family_has_plugin

        registry = MagicMock()
        registry.get.return_value = MagicMock(family="flux2", media="image", id="flux2-dev")
        with patch(
            "backend.engine.sessions.engine_dispatch.is_family_plugin_registered",
            return_value=False,
        ):
            with self.assertRaises(RuntimeError):
                assert_generation_family_has_plugin(
                    "flux2-dev", registry, expected_media="image"
                )

    def test_plugin_upscale_pipeline_if_ready(self) -> None:
        from backend.engine.families._upscale_backbone import (
            UpscalePluginBackbone,
            plugin_upscale_pipeline_if_ready,
        )
        from backend.engine.protocols import FamilyPlugin, FamilySpec

        spec = FamilySpec(family_id="seedvr2", media="image")
        backbone = UpscalePluginBackbone(spec)
        sentinel = object()
        backbone._pipeline = sentinel
        plugin = FamilyPlugin(family_id="seedvr2", spec=spec, backbone=backbone)
        self.assertIs(plugin_upscale_pipeline_if_ready(plugin), sentinel)
        self.assertIsNone(plugin_upscale_pipeline_if_ready(None))

    def test_plugin_audio_generator_if_ready(self) -> None:
        from backend.engine.families._audio_backbone import (
            AudioPluginBackbone,
            plugin_audio_generator_if_ready,
        )
        from backend.engine.protocols import FamilyPlugin, FamilySpec

        spec = FamilySpec(family_id="ace_step", media="audio")
        backbone = AudioPluginBackbone(spec)
        sentinel = object()
        backbone._generator = sentinel
        plugin = FamilyPlugin(family_id="ace_step", spec=spec, backbone=backbone)
        self.assertIs(plugin_audio_generator_if_ready(plugin), sentinel)
        self.assertIsNone(plugin_audio_generator_if_ready(None))

    def test_plugin_backbone_model_if_ready(self) -> None:
        from backend.engine.families._image_backbone import (
            ImagePluginBackbone,
            plugin_backbone_model_if_ready,
        )
        from backend.engine.protocols import FamilyPlugin

        spec = FamilySpec(family_id="flux2", media="image")
        backbone = ImagePluginBackbone(spec)
        sentinel = object()
        backbone._model = sentinel
        plugin = FamilyPlugin(family_id="flux2", spec=spec, backbone=backbone)
        request = MagicMock(adapters=None)

        self.assertIs(plugin_backbone_model_if_ready(plugin, request=request), sentinel)
        self.assertIsNone(plugin_backbone_model_if_ready(None, request=request))
        request.adapters = ["lora-a"]
        self.assertIsNone(plugin_backbone_model_if_ready(plugin, request=request))

    def test_run_video_denoise_delegates(self) -> None:
        from backend.engine.inference.video_denoise import run_video_denoise

        pipeline = MagicMock()
        pipeline.ctx = MagicMock()
        ctx_exec = MagicMock()
        ctx_exec.cancel_token.is_cancelled.return_value = False
        model = MagicMock()
        scheduler = MagicMock(init_noise_sigma=1.0)
        latents = MagicMock()
        sentinel = object()

        with unittest.mock.patch(
            "backend.engine.inference.video_denoise.run_diffusion_denoise",
            return_value=sentinel,
        ) as mock_run:
            out = run_video_denoise(
                pipeline,
                model=model,
                scheduler=scheduler,
                timesteps=[0.9, 0.5],
                latents=latents,
                config=MagicMock(),
                guidance=3.5,
                txt_embeds=MagicMock(),
                neg_embeds=None,
                sigmas=None,
                timestep_embed_schedule=None,
                extra_cond={},
                rope_kw={},
                cfg_renorm=False,
                cfg_renorm_min=0.0,
                ctx_exec=ctx_exec,
                on_progress=None,
                on_log=None,
            )
        self.assertIs(out, sentinel)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertIs(kwargs["model"], model)
        self.assertIs(kwargs["latents"], latents)

    def test_infer_phase_video_create_ctx(self) -> None:
        from backend.engine.pipelines.video_create_phases import VideoCreateRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        resolved.plugin = None
        resolved.exec_ctx = MagicMock(trace=None)
        pipeline = MagicMock()
        schedule = ScheduleState(
            scheduler=MagicMock(),
            timesteps=[1.0],
            sigmas=None,
            scheduler_name="euler",
        )
        video_ctx = VideoCreateRunContext(
            pipeline=pipeline,
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            config=MagicMock(),
            family="wan",
            model_key="wan-2.2",
            version_key=None,
            bundle_root=None,
            model=MagicMock(),
            extra_cond={},
            txt_embeds=MagicMock(),
            neg_embeds=None,
            scheduler=MagicMock(),
            timesteps=[1.0],
            sigmas=None,
            timestep_embed_schedule=None,
            latents=MagicMock(),
            rope_kw={},
            w=512,
            h=512,
            num_frames=81,
            fps=24,
            seed=1,
            steps=20,
            guidance=5.0,
            cfg_renorm=False,
            cfg_renorm_min=0.0,
            mode="video_generate",
        )
        sentinel = object()
        with unittest.mock.patch(
            "backend.engine.pipelines.video_create_phases.execute_video_denoise",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                pipeline=pipeline,
                run_ctx=video_ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once()
        self.assertIs(mock_exec.call_args.args[0], pipeline)

    def test_infer_phase_image_fill_edit_ctx(self) -> None:
        from backend.engine.pipelines.image_fill_edit_phases import ImageFillEditRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        ctx = ImageFillEditRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            config=MagicMock(),
            runtime_contract=MagicMock(),
            family="flux1",
            model_key="flux-fill-controlnet",
            version_key=None,
            bundle_root=None,
            model=MagicMock(),
            extra_cond={},
            txt_embeds=None,
            neg_embeds=None,
            txt_attn_mask=None,
            neg_attn_mask=None,
            pooled_embeds=None,
            neg_pooled_embeds=None,
            encoder_type="clip",
            scheduler=MagicMock(),
            timesteps=[1.0],
            sigmas=None,
            sched_ts=None,
            timestep_embed_schedule=None,
            semantics=MagicMock(),
            latents=MagicMock(),
            w=512,
            h=512,
            lh=64,
            lw=64,
            seed=42,
            steps=28,
            guidance=30.0,
            flux_unpack=MagicMock(),
            flux_pack=MagicMock(),
            preview_mode="none",
            preview_interval=2,
            preview_max_edge=512,
            preview_state={},
        )
        sentinel = MagicMock()
        schedule = ScheduleState(MagicMock(), [1.0], None, "image_fill_edit")
        with unittest.mock.patch(
            "backend.engine.pipelines.image_fill_edit_phases.execute_image_fill_edit_denoise",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                pipeline=ctx.pipeline,
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)

    def test_infer_phase_qwen_image_edit_ctx(self) -> None:
        from backend.engine.families.qwen.edit_util import QwenImageEditRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        ctx = QwenImageEditRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            config=MagicMock(),
            runtime_contract=MagicMock(),
            family="qwen_image",
            model_key="qwen-image-edit",
            version_key=None,
            bundle_root=None,
            model=MagicMock(),
            extra_cond={},
            txt_embeds=None,
            neg_embeds=None,
            txt_attn_mask=None,
            neg_attn_mask=None,
            pooled_embeds=None,
            neg_pooled_embeds=None,
            encoder_type="qwen_image",
            scheduler=MagicMock(),
            timesteps=[1.0],
            sigmas=None,
            sched_ts=None,
            timestep_embed_schedule=None,
            semantics=MagicMock(),
            latents=MagicMock(),
            w=512,
            h=512,
            seed=42,
            steps=20,
            guidance=4.0,
            preview_mode="none",
            preview_interval=2,
            preview_max_edge=512,
            preview_state={},
        )
        sentinel = MagicMock()
        schedule = ScheduleState(MagicMock(), [1.0], None, "qwen_image_edit")
        with unittest.mock.patch(
            "backend.engine.families.qwen.edit_util.execute_qwen_image_edit_denoise",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                pipeline=ctx.pipeline,
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)

    def test_infer_phase_image_edit_ctx(self) -> None:
        from backend.engine.pipelines.image_edit_phases import ImageEditRunContext
        from backend.engine.contracts import FamilyRuntimeContract
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        config = MagicMock()
        ctx = ImageEditRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            config=config,
            runtime_contract=FamilyRuntimeContract(family="flux2", config=config),
            family="flux2",
            model_key="flux2-dev",
            version_key=None,
            bundle_root=None,
            model=MagicMock(),
            extra_cond={},
            txt_embeds=MagicMock(),
            neg_embeds=None,
            txt_attn_mask=None,
            neg_attn_mask=None,
            pooled_embeds=None,
            neg_pooled_embeds=None,
            encoder_type="clip",
            scheduler=MagicMock(),
            timesteps=[1.0],
            sigmas=None,
            sched_ts=None,
            timestep_embed_schedule=None,
            semantics=MagicMock(),
            latents=MagicMock(),
            w=512,
            h=512,
            seed=1,
            steps=4,
            guidance=3.5,
            init_timestep=0,
            fidelity=0.5,
            source_pil=MagicMock(),
            preview_mode="none",
            preview_interval=2,
            preview_max_edge=512,
            preview_decoder="none",
            preview_state={},
            packed_edit=False,
            flux_unpack_edit=None,
            lh_edit=0,
            lw_edit=0,
            edit_conditioning_concat=False,
            structural_output_meta=None,
        )
        sentinel = object()
        schedule = ScheduleState(MagicMock(), [1.0], None, "image_edit")
        with unittest.mock.patch(
            "backend.engine.pipelines.image_edit_phases.execute_image_edit_denoise",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)

    def test_infer_phase_audio_edit_ctx(self) -> None:
        from backend.engine.pipelines.audio_edit_phases import AudioEditRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        ctx = AudioEditRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            model_id="ace-step-xl-sft",
            version_key=None,
            entry=MagicMock(),
            bundle_root=Path("/tmp"),
            config=MagicMock(),
            family="ace_step",
            generator=MagicMock(),
            handler=MagicMock(),
            src_path=Path("/tmp/src.wav"),
            paradigm="flow_matching",
            t0=0.0,
        )
        sentinel = ([], [], MagicMock(), None)
        schedule = ScheduleState(None, [], None, "flow_matching")
        with unittest.mock.patch(
            "backend.engine.pipelines.audio_edit_phases.execute_audio_edit_infer",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)

    def test_infer_phase_upscale_create_ctx(self) -> None:
        from backend.engine.pipelines.upscale_create_phases import UpscaleCreateRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        ctx = UpscaleCreateRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            family="seedvr2",
            model_key="seedvr2",
            version_key=None,
            bundle_root=Path("/tmp"),
            src_path=Path("/tmp/src.png"),
            scale=2,
            seed=None,
            out_path=Path("/tmp/out.png"),
            upscale_pipeline=MagicMock(),
        )
        sentinel = {"ok": True}
        schedule = ScheduleState(None, [], None, "job")
        with unittest.mock.patch(
            "backend.engine.pipelines.upscale_create_phases.execute_upscale_job",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)

    def test_infer_phase_video_upscale_create_ctx(self) -> None:
        from backend.engine.pipelines.video_upscale_create_phases import VideoUpscaleCreateRunContext
        from backend.engine.sessions._phases.infer import infer_phase
        from backend.engine.sessions._phases.schedule import ScheduleState

        resolved = MagicMock()
        ctx = VideoUpscaleCreateRunContext(
            pipeline=MagicMock(),
            request=MagicMock(),
            exec_ctx=MagicMock(),
            entry=MagicMock(),
            family="hunyuan",
            model_key="hunyuan-video-1.5-1080p-sr",
            version_key=None,
            kind="hunyuan_1080p_sr",
        )
        sentinel = ("/tmp/out.mp4", {"model": "hunyuan-video-1.5-1080p-sr"})
        schedule = ScheduleState(None, [], None, "job")
        with unittest.mock.patch(
            "backend.engine.pipelines.video_upscale_create_phases.execute_video_upscale_job",
            return_value=sentinel,
        ) as mock_exec:
            out = infer_phase(
                resolved,
                {},
                schedule,
                runtime_ctx=MagicMock(),
                run_ctx=ctx,
            )
        self.assertIs(out, sentinel)
        mock_exec.assert_called_once_with(ctx)


if __name__ == "__main__":
    unittest.main()
