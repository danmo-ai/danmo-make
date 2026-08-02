"""Unit tests for long-video storyboard planning and parsing."""
from __future__ import annotations

import unittest

from backend.long_video.plan import (
    allocate_shot_durations,
    build_shot_plan,
    build_shot_plan_for_scenes,
)
from backend.engine.families.ltx.long_video_plan import build_long_video_plan, compute_extend_pass_count
from backend.engine.llm.storyboard import (
    apply_storyboard_output_locale,
    build_structured_shots,
    chapter_beats_ready_for_shots,
    coalesce_dual_pairs,
    dual_pairs_from_beats,
    expand_batches_for_plan,
    is_valid_shot_scene_text,
    merge_visual_with_character_anchor,
    compose_keyframe_visual_prompt,
    is_structured_keyframe_visual,
    KEYFRAME_REF_DIVIDER,
    normalize_shot_scene_text,
    normalize_character_anchor,
    parse_anchor_blocks,
    parse_dual_shot_script,
    parse_expand_script,
    parse_plan_script,
    prompt_leads_with_standalone_pronoun,
    prompt_locale,
    resolve_shot_scene_for_index,
    storyboard_prompts_self_contained,
    storyboard_quality_ok,
    storyboard_shot_pairs_ok,
    visual_includes_anchor_appearance,
    _split_beat_marked_lines,
)
from tests.storyboard_brief_cases import SYNOPSIS_BRIEF_CASES, SynopsisBriefCase


class LongVideoPlanTests(unittest.TestCase):
    def test_extend_pass_count_scales_with_target(self) -> None:
        self.assertEqual(compute_extend_pass_count(30, 8, 8), 3)
        self.assertEqual(compute_extend_pass_count(60, 8, 8), 7)
        self.assertEqual(compute_extend_pass_count(90, 8, 8), 11)

    def test_build_plan_segments(self) -> None:
        plan = build_long_video_plan(target_duration_sec=60, initial_duration_sec=8, segment_extend_sec=8)
        self.assertEqual(plan.extend_pass_count, 7)
        self.assertEqual(plan.total_segments, 8)
        self.assertEqual(plan.narrative_budget, "standard")

    def test_build_shot_plan_for_scenes(self) -> None:
        plan = build_shot_plan_for_scenes(scene_count=6, segment_duration_sec=5.0)
        self.assertEqual(plan.shot_count, 6)
        self.assertEqual(len(plan.segment_durations_sec), 6)
        self.assertAlmostEqual(plan.target_duration_sec, 30.0, delta=6.0)

    def test_allocate_shot_durations_soft_budget(self) -> None:
        durations = allocate_shot_durations(
            scene_count=10,
            target_duration_sec=45,
            default_segment_sec=5,
            beat_texts=["远景建立镜头"] + ["沙漠大战决斗长镜头"] * 9,
        )
        self.assertEqual(len(durations), 10)
        self.assertGreater(sum(durations), 30)
        self.assertGreater(durations[1], durations[0])

    def test_build_shot_plan_for_scenes_clamps(self) -> None:
        low = build_shot_plan_for_scenes(scene_count=1, segment_duration_sec=5.0)
        self.assertEqual(low.shot_count, 2)
        high = build_shot_plan_for_scenes(scene_count=40, segment_duration_sec=5.0)
        self.assertEqual(high.shot_count, 24)

    def test_expand_batches_when_many_passes(self) -> None:
        plan = build_long_video_plan(target_duration_sec=90, initial_duration_sec=8, segment_extend_sec=8)
        batches = expand_batches_for_plan(plan)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(c for _, c in batches), plan.extend_pass_count)


class StoryboardParserTests(unittest.TestCase):
    def test_parse_plan_script(self) -> None:
        text = (
            "[Anchor] Red coat detective, neon alley, cool blue rim light.\n"
            "[Beat 1] Opens on rain and footsteps.\n"
            "[Beat 2] She turns toward camera.\n"
        )
        anchor, beats = parse_plan_script(text, expected_beats=2)
        self.assertIn("Red coat", anchor)
        self.assertEqual(len(beats), 2)

    def test_parse_plan_script_pads_short_llm_output(self) -> None:
        text = "[Anchor] A cat in neon city.\n[Beat 1] Cat wakes.\n[Beat 2] Cat walks."
        anchor, beats = parse_plan_script(text, expected_beats=6)
        self.assertEqual(len(beats), 6)
        self.assertEqual(beats[-1], beats[1])

    def test_parse_expand_script(self) -> None:
        text = (
            "[Opening] Anchor scene with slow dolly in, rain ambience.\n"
            "[Segment 1] She walks deeper into the alley, same red coat.\n"
            "[Segment 2] Close-up reaction, thunder rumble.\n"
        )
        opening, segs = parse_expand_script(text, expected_segments=2)
        self.assertTrue(opening)
        self.assertEqual(len(segs), 2)

    def test_parse_expand_script_loose_lines(self) -> None:
        text = (
            "楚人美在豆包建议下向孙悟空发起挑战。\n"
            "孙悟空拔出一根毫毛，毫毛化作分身。\n"
            "楚人美被毫毛一击击败，坠入地府。\n"
        )
        _opening, segs = parse_expand_script(text, expected_segments=4)
        self.assertEqual(len(segs), 4)

    def test_parse_expand_script_uses_fallback_when_empty(self) -> None:
        text = ""
        fallback = ["beat one", "beat two", "beat three"]
        _opening, segs = parse_expand_script(
            text, expected_segments=3, fallback=fallback,
        )
        self.assertEqual(segs, fallback)

    def test_parse_dual_shot_script_chinese(self) -> None:
        text = (
            "[Visual 1] 霓虹雨巷中，穿红风衣的女侦探侧身站立，冷色侧光，35mm 近景。\n"
            "[Motion 1] 镜头缓慢跟拍，她转身向镜头走来，雨滴在地面溅起。\n"
            "[Visual 2] 同一人物到达巷口铁门前，手搭门把，霓虹招牌反光。\n"
            "[Motion 2] 推近至面部特写，她停顿后推门而入。\n"
        )
        pairs = parse_dual_shot_script(text, expected_shots=2)
        self.assertEqual(len(pairs), 2)
        self.assertIn("红风衣", pairs[0][0])
        self.assertIn("跟拍", pairs[0][1])

    def test_coalesce_splits_multi_beat_blob(self) -> None:
        anchor = "赵今麦身着现代简约白T恤，黑色短发"
        blob = (
            f"{anchor}\n"
            "[Beat 1] 赵今麦抬头凝视手机屏幕\n"
            "[Beat 2] 她走向山间石径\n"
            "[Beat 3] 金光从指尖迸发"
        )
        beats = ["赵今麦抬头凝视手机屏幕", "她走向山间石径", "金光从指尖迸发"]
        pairs = coalesce_dual_pairs([(blob, "same motion")] * 3, beats, 3, character_anchor=anchor)
        self.assertEqual(len(pairs), 3)
        self.assertNotEqual(pairs[0][0], pairs[1][0])
        self.assertIn("手机屏幕", pairs[0][0])
        self.assertIn("石径", pairs[1][0])

    def test_build_structured_shots_distinct_per_index(self) -> None:
        beats = ["场景A", "场景B", "场景C"]
        pairs = dual_pairs_from_beats(beats, 3, character_anchor="主角锚点")
        shots = build_structured_shots(
            character_anchor="主角锚点",
            opening_prompt="",
            segment_prompts=[],
            beat_sheet=beats,
            target_duration_sec=15,
            segment_duration_sec=5,
            dual_pairs=pairs,
        )
        visuals = [s["visual_prompt"] for s in shots]
        self.assertEqual(len(set(visuals)), 3)

    def test_split_beat_marked_lines(self) -> None:
        text = "[Beat 1] 第一镜\n[Beat 2] 第二镜"
        parts = _split_beat_marked_lines(text)
        self.assertEqual(len(parts), 2)
        self.assertIn("第一镜", parts[0])

    def test_dual_pairs_from_beats(self) -> None:
        beats = ["楚人美听信豆包", "挑战孙悟空", "毫毛击败", "阎王嘲笑"]
        pairs = dual_pairs_from_beats(beats, 6)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(pairs[0][0], beats[0])
        self.assertEqual(pairs[-1][0], beats[-1])

    def test_storyboard_quality_ok(self) -> None:
        plan = build_long_video_plan(target_duration_sec=30, initial_duration_sec=8, segment_extend_sec=8)
        segs = [
            "She continues down the alley, same red coat, side tracking shot.",
            "Close on face under neon, rain rhythm continues.",
            "She reaches the door, hand on handle, ambient thunder.",
        ]
        beats = ["open rain", "alley walk", "door approach", "enter tension"]
        ok = storyboard_quality_ok(
            character_anchor="A woman in a red coat under neon signs in a wet alley.",
            opening_prompt="Red-coated detective enters rainy neon alley, slow dolly, ambient rain.",
            segment_prompts=segs,
            beat_sheet=beats,
            plan=plan,
        )
        self.assertTrue(ok)


def _shot_count_for(case: SynopsisBriefCase) -> int:
    return build_shot_plan(
        target_duration_sec=case.target_duration_sec,
        segment_duration_sec=case.segment_duration_sec,
    ).shot_count


def _build_shots_from_case(
    case: SynopsisBriefCase,
    *,
    use_broken_expand: bool = False,
) -> tuple[list[dict], list[tuple[str, str]], list[str], str]:
    shot_plan = build_shot_plan(
        target_duration_sec=case.target_duration_sec,
        segment_duration_sec=case.segment_duration_sec,
    )
    shot_count = shot_plan.shot_count
    anchor, beat_sheet = parse_plan_script(case.plan_script, expected_beats=shot_count)

    expand_raw = case.broken_expand_blob if use_broken_expand else case.expand_script
    dual_pairs = parse_dual_shot_script(
        expand_raw,
        expected_shots=shot_count,
        fallback=beat_sheet,
    )
    dual_pairs = coalesce_dual_pairs(
        dual_pairs,
        beat_sheet,
        shot_count,
        character_anchor=anchor,
    )
    from backend.engine.llm.storyboard_cast import parse_character_roster, roster_to_dtos

    roster, style_anchor = parse_character_roster(anchor, locale="zh")
    shots = build_structured_shots(
        character_anchor=anchor,
        opening_prompt="",
        segment_prompts=[],
        beat_sheet=beat_sheet,
        target_duration_sec=case.target_duration_sec,
        segment_duration_sec=case.segment_duration_sec,
        dual_pairs=dual_pairs,
        characters=roster_to_dtos(roster),
        style_anchor=style_anchor,
        locale="zh",
    )
    return shots, dual_pairs, beat_sheet, anchor


def _composed_t2i_for_shot(shot: dict, anchor: str) -> str:
    from backend.engine.llm.storyboard_cast import (
        compose_keyframe_with_cast,
        dtos_to_cast_looks,
        dtos_to_roster,
        infer_shot_cast_looks,
        parse_character_roster,
    )

    roster, style = parse_character_roster(anchor, locale="zh")
    scene = str(shot.get("scene_prompt") or shot.get("visual_prompt", "")).strip()
    cast = dtos_to_cast_looks(shot.get("cast_looks") or [])
    if not cast and roster:
        cast = infer_shot_cast_looks(scene=scene, beat=scene, characters=roster)
    return compose_keyframe_with_cast(
        scene,
        characters=roster,
        cast=cast,
        style_anchor=style,
        locale="zh",
        character_anchor=anchor,
    )


class SynopsisBriefCaseTests(unittest.TestCase):
    def test_all_synopsis_cases_produce_distinct_shots(self) -> None:
        for case in SYNOPSIS_BRIEF_CASES:
            with self.subTest(case=case.name):
                shots, dual_pairs, beat_sheet, anchor = _build_shots_from_case(case)
                shot_count = _shot_count_for(case)
                self.assertEqual(len(shots), shot_count)
                self.assertGreaterEqual(len(anchor.strip()), 12)
                visuals = [s["visual_prompt"] for s in shots]
                motions = [s["motion_prompt"] for s in shots]
                self.assertEqual(len(set(visuals)), shot_count, msg=case.name)
                for i, shot in enumerate(shots):
                    composed = _composed_t2i_for_shot(shot, anchor)
                    self.assertTrue(
                        is_structured_keyframe_visual(composed),
                        msg=f"{case.name} shot {i} t2i not structured",
                    )
                    self.assertTrue(
                        visual_includes_anchor_appearance(composed, anchor),
                        msg=f"{case.name} shot {i} missing anchor appearance",
                    )
                self.assertTrue(
                    storyboard_shot_pairs_ok(
                        dual_pairs,
                        shot_count=shot_count,
                        beat_sheet=beat_sheet,
                    ),
                    msg=case.name,
                )
                for i, (visual, motion) in enumerate(zip(visuals, motions)):
                    self.assertGreaterEqual(len(visual), 6, msg=f"{case.name} shot {i}")
                    self.assertGreaterEqual(len(motion), 6, msg=f"{case.name} shot {i}")

    def test_zhao_jinmai_broken_expand_blob_recovered(self) -> None:
        case = next(c for c in SYNOPSIS_BRIEF_CASES if c.name == "zhao_jinmai_vs_wukong")
        self.assertTrue(case.broken_expand_blob)
        shots, dual_pairs, beat_sheet, anchor = _build_shots_from_case(
            case,
            use_broken_expand=True,
        )
        shot_count = _shot_count_for(case)
        visuals = [s["visual_prompt"] for s in shots]
        self.assertEqual(len(set(visuals)), shot_count)
        self.assertIn("手机", visuals[0])
        self.assertIn("阎罗", visuals[-1])
        for shot in shots:
            composed = _composed_t2i_for_shot(shot, anchor)
            self.assertTrue(is_structured_keyframe_visual(composed))
            self.assertTrue(visual_includes_anchor_appearance(composed, anchor))
        self.assertTrue(
            storyboard_shot_pairs_ok(
                dual_pairs,
                shot_count=shot_count,
                beat_sheet=beat_sheet,
            )
        )

    def test_zhao_jinmai_plan_beats_match_brief_arc(self) -> None:
        case = next(c for c in SYNOPSIS_BRIEF_CASES if c.name == "zhao_jinmai_vs_wukong")
        shot_count = _shot_count_for(case)
        _anchor, beats = parse_plan_script(case.plan_script, expected_beats=shot_count)
        joined = " ".join(beats)
        for keyword in ("豆包", "孙悟空", "毫毛", "地府", "阎罗"):
            self.assertIn(keyword, joined)
        self.assertIn("赵今麦", case.brief)

    def test_parse_plan_anchor_stops_before_beat_tags(self) -> None:
        text = (
            "[Anchor] 赵今麦，素色襦裙，手持竹笛。\n"
            "[Beat 1] 低头看手机。\n"
            "[Beat 2] 走向花果山。\n"
        )
        anchor, beats = parse_plan_script(text, expected_beats=2)
        self.assertNotIn("[Beat", anchor)
        self.assertNotIn("花果山", anchor)
        self.assertEqual(beats[0], "低头看手机。")

    def test_visual1_with_inline_beats_yields_single_static_keyframe_prompts(self) -> None:
        """LLM often puts [Beat 1..N] under [Visual 1]; keyframes must be one static frame each."""
        expand = (
            "[Visual 1] 赵今麦，20 岁，素色襦裙，青竹簪，月白底色，手持竹笛，背景为青石小院。\n"
            "[Beat 1] 赵今麦低头看手机，豆包 App 弹出红色提示。\n"
            "[Beat 2] 她甩开竹笛，大步走向云雾缭绕的花果山。\n"
            "[Beat 3] 孙悟空金箍棒横在身侧，突然出现在近前。\n"
            "[Beat 4] 赵今麦后退一步，竹笛脱手落地。\n"
            "[Beat 5] 孙悟空弹出一根毫毛，毫毛化作细线。\n"
            "[Motion 1] 镜头缓慢推近手机。\n"
            "[Motion 2] 跟拍侧移上山。\n"
            "[Motion 3] 固定机位对峙。\n"
            "[Motion 4] handheld 后退。\n"
            "[Motion 5] 快切毫毛飞出。"
        )
        plan = (
            "[Anchor] 赵今麦，20 岁，素色襦裙，青竹簪，月白底色，手持竹笛，背景为青石小院。\n"
            "[Beat 1] 赵今麦低头看手机。\n"
            "[Beat 2] 她甩开竹笛走向花果山。\n"
            "[Beat 3] 孙悟空出现。\n"
            "[Beat 4] 赵今麦后退。\n"
            "[Beat 5] 毫毛飞出。"
        )
        anchor, beats = parse_plan_script(plan, expected_beats=5)
        pairs = parse_dual_shot_script(expand, expected_shots=5, fallback=beats)
        pairs = coalesce_dual_pairs(pairs, beats, 5, character_anchor=anchor)
        shots = build_structured_shots(
            character_anchor=anchor,
            opening_prompt="",
            segment_prompts=[],
            beat_sheet=beats,
            target_duration_sec=25,
            segment_duration_sec=5,
            dual_pairs=pairs,
        )
        visuals = [s["visual_prompt"] for s in shots]
        for i, visual in enumerate(visuals):
            self.assertNotIn("[Beat", visual, msg=f"shot {i} must not contain beat tags")
            self.assertNotIn("[Visual", visual, msg=f"shot {i} must not contain structural tags")
        self.assertEqual(len(set(visuals)), 5)
        self.assertIn("手机", visuals[0])
        self.assertIn("花果山", visuals[1])
        self.assertIn("毫毛", visuals[-1])
        for visual in visuals:
            self.assertFalse(
                all(k in visual for k in ("[Beat 1]", "[Beat 2]")),
                "single keyframe must not merge multiple beats",
            )

    def test_keyframe_zero_same_rules_as_other_shots(self) -> None:
        """Every keyframe visual must carry shared wardrobe/hair from [Anchor]."""
        beats = ["低头看手机，豆包弹出提示。", "走向花果山。", "孙悟空现身。"]
        anchor = "赵今麦，素色襦裙，青竹簪，手持竹笛。"
        pairs = dual_pairs_from_beats(beats, 3, character_anchor=anchor)
        shots = build_structured_shots(
            character_anchor=anchor,
            opening_prompt="",
            segment_prompts=[],
            beat_sheet=beats,
            target_duration_sec=15,
            segment_duration_sec=5,
            dual_pairs=pairs,
        )
        for shot in shots:
            visual = shot["visual_prompt"]
            self.assertIn("青竹簪", _composed_t2i_for_shot(shot, anchor))
            self.assertNotIn("青竹簪", visual)
            self.assertTrue(visual_includes_anchor_appearance(_composed_t2i_for_shot(shot, anchor), anchor))


class StoryboardLocaleTests(unittest.TestCase):
    def test_prompt_locale_detects_english(self) -> None:
        self.assertEqual(prompt_locale("20-year-old Zhao standing in courtyard"), "en")

    def test_prompt_locale_detects_chinese(self) -> None:
        self.assertEqual(prompt_locale("赵今麦低头看手机，豆包弹出红色提示"), "zh")

    def test_apply_storyboard_output_locale_replaces_english_visual_with_zh_beat(self) -> None:
        beats = [
            "赵今麦低头看手机，豆包弹出提示。",
            "她走向花果山。",
        ]
        shots = [
            {"id": "shot_00", "order": 0, "visual_prompt": "20-year-old Zhao on stone path", "motion_prompt": "slow dolly"},
            {"id": "shot_01", "order": 1, "visual_prompt": "walk to mountain", "motion_prompt": "pan left"},
        ]
        fixed = apply_storyboard_output_locale(shots, beat_sheet=beats, locale="zh")
        self.assertEqual(fixed[0]["visual_prompt"], beats[0])
        self.assertEqual(fixed[1]["visual_prompt"], beats[1])


class StoryboardAppearanceTests(unittest.TestCase):
    def test_compose_structured_keyframe_prompt(self) -> None:
        anchor = (
            "【角色·赵今麦】现代简约白T恤，黑色短发\n"
            "---\n"
            "【画风】写实电影感，35mm浅景深"
        )
        visual = "近景，赵今麦在卧室暖光下盯手机屏幕"
        merged = compose_keyframe_visual_prompt(visual, anchor, locale="zh")
        self.assertIn(KEYFRAME_REF_DIVIDER, merged)
        self.assertIn("【本帧】", merged)
        scene_part, ref_part = merged.split(KEYFRAME_REF_DIVIDER, 1)
        self.assertIn("白T恤", ref_part)
        self.assertIn("盯手机屏幕", scene_part)

    def test_merge_prepends_wardrobe_when_missing(self) -> None:
        anchor = "赵今麦，现代简约白T恤与黑色短发，写实电影感"
        visual = "近景，赵今麦在卧室暖光下盯手机屏幕"
        merged = merge_visual_with_character_anchor(visual, anchor)
        self.assertIn("白T恤", merged.split(KEYFRAME_REF_DIVIDER, 1)[1])
        self.assertIn("盯手机屏幕", merged.split(KEYFRAME_REF_DIVIDER, 1)[0])
        self.assertTrue(is_structured_keyframe_visual(merged))

    def test_merge_skips_when_already_structured(self) -> None:
        anchor = "【角色·赵今麦】素色襦裙，青竹簪"
        visual = (
            "【角色·赵今麦】素色襦裙，青竹簪\n"
            "---\n"
            "【本帧】低头看手机"
        )
        self.assertEqual(merge_visual_with_character_anchor(visual, anchor), visual)

    def test_multi_character_clause_matches_on_screen_cast(self) -> None:
        from backend.engine.llm.storyboard_cast import (
            compose_keyframe_with_cast,
            infer_shot_cast_looks,
            parse_character_roster,
        )

        anchor = (
            "【角色·赵今麦】白T恤，黑色短发\n"
            "---\n"
            "【角色·孙悟空】金色锁子甲，猴毛"
        )
        visual = "中景，孙悟空云端盘坐，指尖拈着毫毛"
        roster, style = parse_character_roster(anchor, locale="zh")
        cast = infer_shot_cast_looks(scene=visual, beat=visual, characters=roster)
        composed = compose_keyframe_with_cast(
            visual,
            characters=roster,
            cast=cast,
            style_anchor=style,
            locale="zh",
        )
        self.assertIn("锁子甲", composed.split(KEYFRAME_REF_DIVIDER, 1)[1])
        self.assertNotIn("白T恤", composed)
        self.assertTrue(composed.startswith("【本帧】") or "孙悟空" in composed)

    def test_normalize_legacy_anchor_to_blocks(self) -> None:
        legacy = "赵今麦，白T恤，黑色短发。孙悟空，金色锁子甲。"
        normalized = normalize_character_anchor(legacy, locale="zh")
        self.assertIn(KEYFRAME_REF_DIVIDER, normalized)
        blocks = parse_anchor_blocks(normalized)
        names = [n for k, n, _ in blocks if k == "character"]
        self.assertIn("赵今麦", names)
        self.assertIn("孙悟空", names)


class StoryboardSceneTextTests(unittest.TestCase):
    def test_rejects_instruction_leakage(self) -> None:
        garbage = "【本帧】必须点名可见角色，输出格式为..."
        self.assertFalse(is_valid_shot_scene_text(garbage))
        self.assertFalse(is_valid_shot_scene_text("必须点名可见角色"))

    def test_accepts_structured_beat(self) -> None:
        beat = "【远景】晨曦下，赵今麦站在木屋前，手持手机"
        self.assertTrue(is_valid_shot_scene_text(beat))

    def test_chapter_beats_ready(self) -> None:
        beats = [
            "【远景】晨曦下，赵今麦站在木屋前",
            "【中景】木屋内，赵今麦坐在桌前看手机",
            "【特写】赵今麦凝视着手机屏幕上的胜率提示",
        ]
        self.assertTrue(chapter_beats_ready_for_shots(beats))
        self.assertFalse(chapter_beats_ready_for_shots(["【本帧】必须点名"] * 3))

    def test_normalize_prefers_clean_beat_over_garbage(self) -> None:
        garbage = "【本帧】必须点名可见角色\n【远景】云雾山顶，赵今麦穿红薄外套"
        beat = "【远景】云雾山顶，赵今麦穿红薄外套"
        self.assertEqual(normalize_shot_scene_text(garbage, fallback=beat), beat)

    def test_coalesce_falls_back_when_expand_duplicated(self) -> None:
        beats = [
            "【远景】晨曦下，赵今麦站在木屋前",
            "【中景】木屋内，赵今麦坐在桌前",
            "【特写】赵今麦凝视手机屏幕",
        ]
        instruction = "【本帧】必须点名可见角色"
        dup_pairs = [(instruction, instruction)] * 3
        out = coalesce_dual_pairs(dup_pairs, beats, 3)
        visuals = [v for v, _ in out]
        self.assertEqual(len(set(visuals)), 3)
        self.assertTrue(all(is_valid_shot_scene_text(v) for v in visuals))

    def test_build_structured_shots_from_beats(self) -> None:
        beats = [
            "【远景】晨曦下，赵今麦站在木屋前",
            "【中景】木屋内，赵今麦坐在桌前",
            "【特写】赵今麦凝视手机屏幕",
        ]
        pairs = dual_pairs_from_beats(beats, 3)
        shots = build_structured_shots(
            character_anchor="【角色·赵今麦·日常】黑色长发，白色T恤",
            opening_prompt=beats[0],
            segment_prompts=beats[1:],
            beat_sheet=beats,
            target_duration_sec=15,
            segment_duration_sec=5,
            dual_pairs=pairs,
        )
        self.assertEqual(len(shots), 3)
        scenes = [str(s["scene_prompt"]) for s in shots]
        self.assertEqual(len(set(scenes)), 3)
        self.assertTrue(all(is_valid_shot_scene_text(s) for s in scenes))


class StoryboardPronounQualityTests(unittest.TestCase):
    def test_detects_leading_pronoun(self) -> None:
        self.assertTrue(prompt_leads_with_standalone_pronoun("她走向花果山"))
        self.assertTrue(prompt_leads_with_standalone_pronoun("She walks into the alley"))
        self.assertFalse(prompt_leads_with_standalone_pronoun("赵今麦走向花果山"))

    def test_shot_pairs_fail_when_motion_starts_with_pronoun(self) -> None:
        pairs = [("赵今麦看手机", "她转身离开"), ("孙悟空现身", "金箍棒落地")]
        self.assertFalse(storyboard_prompts_self_contained(pairs, 2))
        self.assertFalse(
            storyboard_shot_pairs_ok(
                pairs,
                shot_count=2,
                beat_sheet=["赵今麦看手机", "孙悟空现身"],
            )
        )


if __name__ == "__main__":
    unittest.main()
