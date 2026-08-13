"""Register family plugins at process startup."""

from __future__ import annotations

_bootstrapped = False


def bootstrap_family_plugins() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    from backend.engine.families.ace_step.plugin import register_ace_step_plugin
    from backend.engine.families.diffrhythm.plugin import register_diffrhythm_plugin
    from backend.engine.families.cogview4.plugin import register_cogview4_plugin
    from backend.engine.families.ernie_image.plugin import register_ernie_image_plugin
    from backend.engine.families.fibo.plugin import register_fibo_plugin
    from backend.engine.families.flux1.plugin import register_flux1_plugin
    from backend.engine.families.flux2.plugin import register_flux2_plugin
    from backend.engine.families.hunyuan.plugin import register_hunyuan_plugin
    from backend.engine.families.hidream_o1.plugin import register_hidream_o1_plugin
    from backend.engine.families.step1x_edit.plugin import register_step1x_edit_plugin
    from backend.engine.families.ltx.plugin import register_ltx_plugin
    from backend.engine.families.ltx25.plugin import register_ltx25_plugin
    from backend.engine.families.longcat.plugin import register_longcat_plugin
    from backend.engine.families.longcat_avatar.plugin import register_longcat_avatar_plugin
    from backend.engine.families.minimax_h3.plugin import register_minimax_h3_plugin
    from backend.engine.families.qwen.plugin import register_qwen_image_plugin
    from backend.engine.families.seedvr2.plugin import register_seedvr2_plugin
    from backend.engine.families.esrgan.plugin import register_esrgan_plugin
    from backend.engine.families.wan.plugin import register_wan_plugin
    from backend.engine.families.boogu.plugin import register_boogu_image_plugin
    from backend.engine.families.z_image.plugin import register_z_image_plugin

    register_flux2_plugin()
    register_z_image_plugin()
    register_qwen_image_plugin()
    register_flux1_plugin()
    register_fibo_plugin()
    register_ernie_image_plugin()
    register_cogview4_plugin()
    register_wan_plugin()
    register_ltx_plugin()
    register_ltx25_plugin()
    register_longcat_plugin()
    register_longcat_avatar_plugin()
    register_minimax_h3_plugin()
    register_hunyuan_plugin()
    register_hidream_o1_plugin()
    register_step1x_edit_plugin()
    register_boogu_image_plugin()
    register_seedvr2_plugin()
    register_esrgan_plugin()
    register_ace_step_plugin()
    register_diffrhythm_plugin()
    _bootstrapped = True
