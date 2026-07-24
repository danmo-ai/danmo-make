<template>
  <section class="lv-scene-workshop lv-section">
    <div v-if="scriptSourceLabel || scenes.length" class="lv-scene-workshop__source">
      <span v-if="scriptSourceLabel" class="lv-scene-workshop__source-text">{{ scriptSourceLabel }}</span>
      <div class="lv-scene-workshop__source-actions">
        <div
          v-if="scenes.length && totalLookCount > 0"
          class="lv-scene-workshop__progress"
          :title="$tt('video.longVideoSceneReadyStat', { ready: readyRefCount, total: totalLookCount })"
        >
          <div class="lv-scene-workshop__progress-track">
            <div
              class="lv-scene-workshop__progress-fill"
              :style="{ width: `${Math.round((readyRefCount / totalLookCount) * 100)}%` }"
            />
          </div>
          <span class="lv-scene-workshop__stat">
            {{ $tt('video.longVideoSceneReadyStat', { ready: readyRefCount, total: totalLookCount }) }}
          </span>
        </div>
        <DqButton
          v-if="missingRefCount > 0"
          size="sm"
          type="default"
          :loading="batchGenerating"
          :disabled="batchGenerating || refGeneratingKey != null"
          @click="emit('batch-generate-refs')"
        >
          {{ $tt('video.longVideoSceneRefBatch', { n: missingRefCount }) }}
        </DqButton>
        <DqButton v-if="scriptSourceLabel" size="sm" type="text" @click="emit('go-to-script')">
          {{ $tt('video.longVideoSceneEditScript') }}
        </DqButton>
      </div>
    </div>

    <div v-if="!scenes.length" class="lv-scene-workshop__empty">
      <div class="lv-scene-workshop__empty-card">
        <p class="lv-scene-workshop__empty-text">{{ $tt('video.longVideoSceneRosterEmpty') }}</p>
        <DqButton type="primary" @click="onAddScene">
          <DqIcon :size="12"><Plus /></DqIcon>
          {{ $tt('video.longVideoSceneAdd') }}
        </DqButton>
      </div>
    </div>

    <div v-else class="lv-scene-workshop__body">
      <nav class="lv-scene-workshop__sidebar" :aria-label="$tt('video.longVideoSceneSidebarAria')">
        <button
          v-for="(sc, si) in scenes"
          :key="sc.id"
          type="button"
          class="lv-scene-workshop__scene"
          :class="{ 'lv-scene-workshop__scene--active': selectedIndex === si }"
          @click="selectedIndex = si"
        >
          <div
            class="lv-scene-workshop__scene-thumb"
            :class="{ 'lv-scene-workshop__scene-thumb--empty': !primaryRefAsset(sc) }"
          >
            <img v-if="primaryRefAsset(sc)" :src="refUrl(primaryRefAsset(sc)!)" alt="" />
            <span v-else class="lv-scene-workshop__scene-initial">{{ sceneInitial(sc.name) }}</span>
          </div>
          <div class="lv-scene-workshop__scene-info">
            <span class="lv-scene-workshop__scene-name">{{ sc.name.trim() || $tt('video.longVideoSceneUntitled') }}</span>
            <span class="lv-scene-workshop__scene-meta">
              {{ $tt('video.longVideoSceneVariantCount', { n: sc.looks.length }) }}
              ·
              {{ $tt('video.longVideoSceneRefsReady', { n: refCountForScene(sc) }) }}
            </span>
          </div>
          <span
            v-if="!sceneRefsComplete(sc)"
            class="lv-scene-workshop__scene-warn"
            :title="$tt('video.longVideoSceneRefMissing')"
          />
        </button>
        <button type="button" class="lv-scene-workshop__scene lv-scene-workshop__scene--add" @click="onAddScene">
          <span class="lv-scene-workshop__scene-add-icon" aria-hidden="true">
            <DqIcon :size="14"><Plus /></DqIcon>
          </span>
          <span>{{ $tt('video.longVideoSceneAdd') }}</span>
        </button>
      </nav>

      <div v-if="selectedScene" class="lv-scene-workshop__detail">
        <header class="lv-scene-workshop__detail-head">
          <div class="lv-scene-workshop__detail-identity">
            <div
              class="lv-scene-workshop__detail-avatar"
              :class="{ 'lv-scene-workshop__detail-avatar--photo': Boolean(primaryRefAsset(selectedScene)) }"
            >
              <img
                v-if="primaryRefAsset(selectedScene)"
                :src="refUrl(primaryRefAsset(selectedScene)!)"
                alt=""
              />
              <span v-else>{{ sceneInitial(selectedScene.name) }}</span>
            </div>
            <div class="lv-scene-workshop__detail-copy">
              <DqInput
                :model-value="selectedScene.name"
                size="sm"
                class="lv-scene-workshop__detail-name"
                :placeholder="$tt('video.longVideoSceneNamePh')"
                @update:model-value="onSceneName(selectedIndex, $event)"
              />
              <div class="lv-scene-workshop__detail-meta">
                <span class="lv-scene-workshop__detail-chip">
                  {{ $tt('video.longVideoSceneVariantCount', { n: selectedScene.looks.length }) }}
                </span>
                <span
                  class="lv-scene-workshop__detail-chip"
                  :class="{ 'is-complete': sceneRefsComplete(selectedScene) }"
                >
                  {{ $tt('video.longVideoSceneRefsReady', { n: refCountForScene(selectedScene) }) }}
                </span>
              </div>
            </div>
          </div>
          <div class="lv-scene-workshop__detail-actions">
            <DqButton type="primary" @click="addLook(selectedIndex)">
              <DqIcon :size="12"><Plus /></DqIcon>
              {{ $tt('video.longVideoSceneAddVariant') }}
            </DqButton>
            <DqIconButton
              type="text"
              size="sm"
              class="lv-scene-workshop__detail-delete"
              :label="$tt('video.longVideoSceneRemove')"
              @click="removeScene(selectedIndex)"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
            </DqIconButton>
          </div>
        </header>

        <div class="lv-scene-workshop__looks">
          <article
            v-for="(lk, li) in selectedScene.looks"
            :key="lk.id"
            class="lv-scene-workshop__look-card"
            :class="{ 'lv-scene-workshop__look-card--ready': Boolean(lk.reference_asset_id) }"
          >
            <div
              class="lv-scene-workshop__look-ref"
              :class="{ 'is-empty': !lk.reference_asset_id, 'is-loading': refGeneratingKey === refKey(selectedIndex, li) }"
            >
              <img v-if="lk.reference_asset_id" :src="refUrl(lk.reference_asset_id)" alt="" />
              <div v-else class="lv-scene-workshop__look-ref-empty">
                <DqIcon :size="22"><PictureFilled /></DqIcon>
                <span>{{ $tt('video.longVideoSceneRefMissing') }}</span>
              </div>
              <div class="lv-scene-workshop__look-badges">
                <span
                  class="lv-scene-workshop__look-badge"
                  :class="lk.reference_asset_id ? 'is-ready' : 'is-pending'"
                >
                  {{ lk.reference_asset_id ? $tt('video.longVideoSceneRefReady') : $tt('video.longVideoSceneRefMissing') }}
                </span>
              </div>
              <div
                v-if="refGeneratingKey === refKey(selectedIndex, li)"
                class="lv-scene-workshop__look-loading"
                aria-live="polite"
              >
                <DqIcon :size="20"><Loading /></DqIcon>
              </div>
              <div v-if="lk.reference_asset_id" class="lv-scene-workshop__look-ref-overlay">
                <button
                  type="button"
                  class="lv-scene-workshop__look-overlay-btn"
                  :disabled="batchGenerating || refGeneratingKey != null"
                  @click="emit('pick-ref-gallery', selectedIndex, li)"
                >
                  {{ $tt('video.longVideoSceneRefImport') }}
                </button>
                <button
                  type="button"
                  class="lv-scene-workshop__look-overlay-btn lv-scene-workshop__look-overlay-btn--danger"
                  :disabled="batchGenerating || refGeneratingKey != null"
                  @click="emit('clear-ref', selectedIndex, li)"
                >
                  {{ $tt('video.longVideoClearShort') }}
                </button>
              </div>
            </div>

            <div class="lv-scene-workshop__look-body">
              <label class="lv-scene-workshop__field">
                <span class="lv-scene-workshop__field-label">{{ $tt('video.longVideoSceneVariantPh') }}</span>
                <DqInput
                  :model-value="lk.label"
                  size="sm"
                  class="lv-scene-workshop__look-label"
                  :placeholder="formatSceneLookOptionLabel(lk, uiLocale)"
                  @update:model-value="onLookLabel(selectedIndex, li, $event)"
                />
              </label>
              <label class="lv-scene-workshop__field">
                <span class="lv-scene-workshop__field-label">{{ $tt('video.longVideoSceneBodyPh') }}</span>
                <DqInput
                  :model-value="lk.body"
                  type="textarea"
                  :rows="3"
                  size="sm"
                  class="lv-scene-workshop__look-desc"
                  :placeholder="$tt('video.longVideoSceneBodyPh')"
                  @update:model-value="onLookBody(selectedIndex, li, $event)"
                />
              </label>
              <label v-if="lk.reference_asset_id || lk.vision_description" class="lv-scene-workshop__field">
                <span class="lv-scene-workshop__field-label">{{ $tt('video.longVideoSceneVisionDescriptionPh') }}</span>
                <DqInput
                  :model-value="lk.vision_description || ''"
                  type="textarea"
                  :rows="2"
                  size="sm"
                  class="lv-scene-workshop__look-vision"
                  :placeholder="$tt('video.longVideoSceneVisionDescriptionExamplePh')"
                  @update:model-value="onLookVisionDescription(selectedIndex, li, $event)"
                />
              </label>
            </div>

            <footer class="lv-scene-workshop__look-foot">
              <DqButton
                size="sm"
                type="primary"
                :loading="refGeneratingKey === refKey(selectedIndex, li)"
                :disabled="batchGenerating || refGeneratingKey != null"
                @click="emit('generate-ref', selectedIndex, li)"
              >
                {{ $tt('video.longVideoSceneRefGenerate') }}
              </DqButton>
              <DqButton
                v-if="!lk.reference_asset_id"
                size="sm"
                type="default"
                :disabled="batchGenerating || refGeneratingKey != null"
                @click="emit('pick-ref-gallery', selectedIndex, li)"
              >
                {{ $tt('video.longVideoSceneRefImport') }}
              </DqButton>
              <DqButton
                v-if="lk.reference_asset_id"
                size="sm"
                type="text"
                :loading="visionBackfillKey === refKey(selectedIndex, li)"
                @click="emit('vision-backfill', selectedIndex, li)"
              >
                {{ $tt('video.longVideoSceneVisionBackfill') }}
              </DqButton>
              <DqIconButton
                v-if="selectedScene.looks.length > 1"
                type="text"
                size="xs"
                class="lv-scene-workshop__look-remove"
                :label="$tt('common.delete')"
                @click="removeLook(selectedIndex, li)"
              >
                <DqIcon :size="12"><Close /></DqIcon>
              </DqIconButton>
            </footer>
            <LongVideoEntityRelatedTasks
              v-if="projectId"
              :project-id="projectId"
              phase="scene_ref"
              :match="{ scene_id: selectedScene.id, scene_look_id: lk.id }"
              :label="$tt('video.longVideoRelatedTasksSceneRef')"
              :refresh-token="refGeneratingKey"
            />
          </article>
        </div>
      </div>
    </div>

    <footer v-if="scriptParsed && scenes.length" class="lv-scene-workshop__next">
      <div class="lv-scene-workshop__next-copy">
        <span class="lv-scene-workshop__next-kicker">{{ $tt('video.longVideoSceneNextTitle') }}</span>
        <p class="lv-scene-workshop__next-text">{{ $tt('video.longVideoSceneNextStoryboard') }}</p>
      </div>
      <DqButton type="primary" @click="emit('go-to-storyboard')">
        {{ $tt('video.longVideoCastNextStoryboardBtn') }}
      </DqButton>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Delete, Loading, PictureFilled, Plus } from '@danqing/dq-shell';
import LongVideoEntityRelatedTasks from './LongVideoEntityRelatedTasks.vue';
import { api } from '@/utils/api';
import type { LongVideoScene } from '@/types';
import {
  createSceneEntry,
  createSceneLookEntry,
  formatSceneLookOptionLabel,
  looksMissingSceneReference,
} from '@/utils/longVideoProject';

const props = defineProps<{
  scenes: LongVideoScene[];
  scriptParsed?: boolean;
  projectId?: string;
  refGeneratingKey?: string | null;
  visionBackfillKey?: string | null;
  batchGenerating?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:scenes', value: LongVideoScene[]): void;
  (e: 'generate-ref', sceneIndex: number, lookIndex: number): void;
  (e: 'pick-ref-gallery', sceneIndex: number, lookIndex: number): void;
  (e: 'clear-ref', sceneIndex: number, lookIndex: number): void;
  (e: 'vision-backfill', sceneIndex: number, lookIndex: number): void;
  (e: 'batch-generate-refs'): void;
  (e: 'go-to-storyboard'): void;
  (e: 'go-to-script'): void;
}>();

const { t: $tt, locale } = useI18n();
const uiLocale = computed(() => (locale.value.startsWith('zh') ? 'zh' : 'en'));
const selectedIndex = ref(0);

const missingRefCount = computed(() => looksMissingSceneReference(props.scenes).length);
const totalLookCount = computed(() =>
  props.scenes.reduce((sum, sc) => sum + sc.looks.length, 0),
);
const readyRefCount = computed(() =>
  props.scenes.reduce(
    (sum, sc) => sum + sc.looks.filter((lk) => Boolean(lk.reference_asset_id)).length,
    0,
  ),
);
const selectedScene = computed(() => props.scenes[selectedIndex.value] ?? null);

const scriptSourceLabel = computed(() => {
  if (!props.scriptParsed) return '';
  const n = props.scenes.length;
  const looks = totalLookCount.value;
  if (n > 0) {
    return $tt('video.longVideoSceneFromScript', { n, looks });
  }
  return $tt('video.longVideoSceneFromScriptEmpty');
});

watch(
  () => props.scenes.length,
  (n) => {
    if (selectedIndex.value >= n) selectedIndex.value = Math.max(0, n - 1);
  },
);

function refKey(si: number, li: number) {
  return `${si}-${li}`;
}

function refUrl(assetId: string) {
  return api.gallery.getImageUrl(`asset:${assetId}`);
}

function sceneInitial(name: string) {
  return (name.trim()[0] || '?').toUpperCase();
}

function primaryRefAsset(sc: LongVideoScene) {
  const lk = sc.looks.find((l) => l.id === sc.default_look_id) || sc.looks[0];
  return lk?.reference_asset_id;
}

function refCountForScene(sc: LongVideoScene) {
  return sc.looks.filter((lk) => Boolean(lk.reference_asset_id)).length;
}

function sceneRefsComplete(sc: LongVideoScene) {
  return sc.looks.length > 0 && sc.looks.every((lk) => Boolean(lk.reference_asset_id));
}

function patchScenes(next: LongVideoScene[]) {
  emit('update:scenes', next);
}

function onAddScene() {
  const loc = locale.value.startsWith('zh') ? 'zh' : 'en';
  const entry = createSceneEntry(loc === 'zh' ? '新场景' : 'New scene', loc);
  patchScenes([...props.scenes, entry]);
  selectedIndex.value = props.scenes.length;
}

function removeScene(si: number) {
  const next = props.scenes.filter((_, i) => i !== si);
  patchScenes(next);
  if (selectedIndex.value >= next.length) {
    selectedIndex.value = Math.max(0, next.length - 1);
  }
}

function onSceneName(si: number, name: string) {
  patchScenes(props.scenes.map((sc, i) => (i === si ? { ...sc, name } : sc)));
}

function addLook(si: number) {
  const sc = props.scenes[si];
  if (!sc) return;
  const loc = locale.value.startsWith('zh') ? 'zh' : 'en';
  const lk = createSceneLookEntry(sc.name, loc === 'zh' ? '变体' : 'variant', loc);
  patchScenes(props.scenes.map((s, i) => (i === si ? { ...s, looks: [...s.looks, lk] } : s)));
}

function removeLook(si: number, li: number) {
  patchScenes(
    props.scenes.map((sc, i) => {
      if (i !== si) return sc;
      const looks = sc.looks.filter((_, j) => j !== li);
      const default_look_id =
        sc.default_look_id === sc.looks[li]?.id ? looks[0]?.id ?? '' : sc.default_look_id;
      return { ...sc, looks, default_look_id };
    }),
  );
}

function onLookLabel(si: number, li: number, label: string) {
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) => (j === li ? { ...lk, label } : lk)),
          },
    ),
  );
}

function onLookBody(si: number, li: number, body: string) {
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) =>
              j === li ? { ...lk, body, environment_prompt: undefined } : lk,
            ),
          },
    ),
  );
}

function onLookVisionDescription(si: number, li: number, visionDescription: string) {
  const trimmed = visionDescription.trim();
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) => {
              if (j !== li) return lk;
              const next = { ...lk, vision_description: trimmed || undefined };
              if (!trimmed) delete next.vision_description;
              return next;
            }),
          },
    ),
  );
}
</script>

<style scoped>
.lv-scene-workshop {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 16px 16px;
  width: 100%;
  box-sizing: border-box;
}

.lv-scene-workshop__source {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  padding-bottom: 10px;
  border-bottom: 0.5px solid var(--dq-border-subtle);
}

.lv-scene-workshop__source-text {
  flex: 1;
  min-width: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.lv-scene-workshop__source-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.lv-scene-workshop__progress {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 140px;
}

.lv-scene-workshop__progress-track {
  width: 72px;
  height: 5px;
  border-radius: 999px;
  overflow: hidden;
  background: color-mix(in srgb, var(--dq-fill-control) 70%, transparent);
}

.lv-scene-workshop__progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--dq-accent) 85%, white),
    var(--dq-accent)
  );
  transition: width 0.25s ease;
}

.lv-scene-workshop__stat {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  padding: 4px 10px;
  border-radius: 999px;
  color: var(--dq-label-secondary);
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 22%, transparent);
  white-space: nowrap;
}

.lv-scene-workshop__empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 240px;
}

.lv-scene-workshop__empty-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  padding: 36px 40px;
  border-radius: 14px;
  border: 1px dashed color-mix(in srgb, var(--dq-accent) 25%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 45%, transparent);
  box-shadow: inset 0 1px 0 color-mix(in srgb, white 4%, transparent);
}

.lv-scene-workshop__empty-text {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  text-align: center;
  max-width: 280px;
}

.lv-scene-workshop__body {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 12px;
  overflow: hidden;
}

.lv-scene-workshop__sidebar {
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-surface-elevated) 55%, transparent);
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  box-shadow: inset 0 1px 0 color-mix(in srgb, white 4%, transparent);
}

.lv-scene-workshop__scene {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 9px 10px 9px 12px;
  border: 0.5px solid transparent;
  border-radius: 11px;
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
}

.lv-scene-workshop__scene::before {
  content: '';
  position: absolute;
  left: 4px;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 999px;
  background: transparent;
  transition: background 0.15s;
}

.lv-scene-workshop__scene:hover {
  background: color-mix(in srgb, var(--dq-fill-control) 65%, transparent);
}

.lv-scene-workshop__scene--active {
  background: color-mix(in srgb, var(--dq-accent) 11%, transparent);
  border-color: color-mix(in srgb, var(--dq-accent) 32%, transparent);
  box-shadow: 0 2px 10px color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lv-scene-workshop__scene--active::before {
  background: var(--dq-accent);
}

.lv-scene-workshop__scene--add {
  justify-content: center;
  padding-left: 10px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  border-style: dashed;
  border-color: var(--dq-border-subtle);
  margin-top: 4px;
}

.lv-scene-workshop__scene--add::before {
  display: none;
}

.lv-scene-workshop__scene-thumb {
  width: 52px;
  height: 36px;
  flex-shrink: 0;
  border-radius: 8px;
  overflow: hidden;
  background: color-mix(in srgb, var(--dq-surface-base) 80%, #000);
  border: 0.5px solid color-mix(in srgb, white 8%, var(--dq-border-subtle));
  box-shadow: 0 2px 8px color-mix(in srgb, black 25%, transparent);
}

.lv-scene-workshop__scene-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-scene-workshop__scene-thumb--empty {
  display: flex;
  align-items: center;
  justify-content: center;
  background: repeating-linear-gradient(
    -45deg,
    transparent,
    transparent 5px,
    color-mix(in srgb, var(--dq-label-tertiary) 7%, transparent) 5px,
    color-mix(in srgb, var(--dq-label-tertiary) 7%, transparent) 6px
  );
}

.lv-scene-workshop__scene-initial {
  font-size: var(--dq-font-size-body);
  font-weight: 700;
  color: var(--dq-label-secondary);
}

.lv-scene-workshop__scene-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.lv-scene-workshop__scene-name {
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  color: var(--dq-label-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.lv-scene-workshop__scene-meta {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-tertiary);
}

.lv-scene-workshop__scene-warn {
  width: 8px;
  height: 8px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--dq-warning, #e6a817);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-warning, #e6a817) 28%, transparent);
}

.lv-scene-workshop__scene-add-icon {
  display: flex;
}

.lv-scene-workshop__detail {
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 0;
  min-width: 0;
  width: 100%;
}

.lv-scene-workshop__detail-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  padding: 10px 12px;
  border-radius: 12px;
  background:
    linear-gradient(
      135deg,
      color-mix(in srgb, var(--dq-accent) 9%, var(--dq-surface-elevated)) 0%,
      color-mix(in srgb, var(--dq-surface-elevated) 55%, transparent) 52%
    );
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 18%, var(--dq-border-subtle));
  box-shadow:
    inset 3px 0 0 var(--dq-accent),
    inset 0 1px 0 color-mix(in srgb, white 5%, transparent),
    0 6px 20px color-mix(in srgb, black 10%, transparent);
  flex-shrink: 0;
}

.lv-scene-workshop__detail-identity {
  display: flex;
  align-items: center;
  gap: 14px;
  flex: 1;
  min-width: 0;
}

.lv-scene-workshop__detail-avatar {
  width: 60px;
  height: 42px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  overflow: hidden;
  font-size: var(--dq-font-size-title);
  font-weight: 700;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 32%, transparent);
  box-shadow:
    0 0 0 2px color-mix(in srgb, var(--dq-accent) 12%, transparent),
    0 4px 14px color-mix(in srgb, black 22%, transparent);
}

.lv-scene-workshop__detail-avatar--photo {
  padding: 0;
  background: #000;
  border-color: color-mix(in srgb, white 10%, var(--dq-border-subtle));
}

.lv-scene-workshop__detail-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-scene-workshop__detail-copy {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-scene-workshop__detail-name {
  width: 100%;
  max-width: 420px;
}

.lv-scene-workshop__detail-name :deep(.dq-input) {
  width: 100%;
  padding: 0;
  font-size: var(--dq-font-size-display);
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.25;
  color: var(--dq-label-primary);
  background: transparent;
  border: none;
  box-shadow: none;
  border-radius: 8px;
  transition: background 0.15s, box-shadow 0.15s;
}

.lv-scene-workshop__detail-name :deep(.dq-input::placeholder) {
  color: var(--dq-label-tertiary);
  font-weight: 500;
}

.lv-scene-workshop__detail-name :deep(.dq-input:hover:not(:disabled)) {
  background: color-mix(in srgb, var(--dq-fill-control) 35%, transparent);
}

.lv-scene-workshop__detail-name :deep(.dq-input:focus) {
  outline: none;
  background: color-mix(in srgb, var(--dq-fill-control) 50%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 22%, transparent);
  padding: 4px 10px;
}

.lv-scene-workshop__detail-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.lv-scene-workshop__detail-chip {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--dq-label-secondary);
  background: color-mix(in srgb, var(--dq-fill-control) 55%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-scene-workshop__detail-chip.is-complete {
  color: color-mix(in srgb, var(--dq-success) 85%, white);
  background: color-mix(in srgb, var(--dq-success) 12%, transparent);
  border-color: color-mix(in srgb, var(--dq-success) 35%, transparent);
}

.lv-scene-workshop__detail-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  padding-left: 12px;
  border-left: 0.5px solid color-mix(in srgb, var(--dq-border-subtle) 80%, transparent);
}

.lv-scene-workshop__detail-delete {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  color: var(--dq-label-tertiary);
  background: color-mix(in srgb, var(--dq-fill-control) 40%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-scene-workshop__detail-delete:hover {
  color: var(--dq-danger);
  background: color-mix(in srgb, var(--dq-danger) 10%, transparent);
  border-color: color-mix(in srgb, var(--dq-danger) 35%, transparent);
}

.lv-scene-workshop__looks {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  align-content: start;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.lv-scene-workshop__look-card {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-surface-elevated) 42%, transparent);
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  box-shadow:
    0 1px 0 color-mix(in srgb, white 4%, transparent),
    0 8px 24px color-mix(in srgb, black 12%, transparent);
  transition: border-color 0.18s, box-shadow 0.18s, transform 0.18s;
}

.lv-scene-workshop__look-card:hover {
  transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--dq-accent) 28%, var(--dq-border-subtle));
  box-shadow:
    0 1px 0 color-mix(in srgb, white 5%, transparent),
    0 12px 28px color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.lv-scene-workshop__look-card--ready {
  border-color: color-mix(in srgb, var(--dq-success) 22%, var(--dq-border-subtle));
}

.lv-scene-workshop__look-ref {
  position: relative;
  aspect-ratio: 16 / 9;
  max-height: 220px;
  overflow: hidden;
  background: #080808;
  border-bottom: 0.5px solid var(--dq-border-subtle);
}

.lv-scene-workshop__look-ref img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-scene-workshop__look-badges {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  pointer-events: none;
}

.lv-scene-workshop__look-badge {
  padding: 3px 8px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: 0.01em;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 0.5px solid rgba(255, 255, 255, 0.12);
}

.lv-scene-workshop__look-badge.is-pending {
  color: #ffd89a;
  background: color-mix(in srgb, var(--dq-warning, #e6a817) 55%, rgba(0, 0, 0, 0.5));
}

.lv-scene-workshop__look-badge.is-ready {
  color: #b8f5c8;
  background: color-mix(in srgb, var(--dq-success) 45%, rgba(0, 0, 0, 0.5));
}

.lv-scene-workshop__look-ref-empty {
  height: 100%;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  background: repeating-linear-gradient(
    -45deg,
    transparent,
    transparent 8px,
    color-mix(in srgb, var(--dq-label-tertiary) 5%, transparent) 8px,
    color-mix(in srgb, var(--dq-label-tertiary) 5%, transparent) 9px
  );
}

.lv-scene-workshop__look-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, black 50%, transparent);
  color: white;
}

.lv-scene-workshop__look-ref-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  opacity: 0;
  background: color-mix(in srgb, black 58%, transparent);
  transition: opacity 0.18s;
}

.lv-scene-workshop__look-ref:hover .lv-scene-workshop__look-ref-overlay {
  opacity: 1;
}

.lv-scene-workshop__look-overlay-btn {
  padding: 6px 12px;
  border-radius: 8px;
  border: 0.5px solid rgba(255, 255, 255, 0.22);
  background: rgba(255, 255, 255, 0.1);
  color: white;
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s;
}

.lv-scene-workshop__look-overlay-btn:hover {
  background: rgba(255, 255, 255, 0.18);
}

.lv-scene-workshop__look-overlay-btn--danger {
  border-color: color-mix(in srgb, var(--dq-danger) 50%, transparent);
  color: #ffc9c9;
}

.lv-scene-workshop__look-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 14px 0;
}

.lv-scene-workshop__field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 0;
}

.lv-scene-workshop__field-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.lv-scene-workshop__look-desc :deep(textarea) {
  min-height: 76px;
  resize: vertical;
}

.lv-scene-workshop__look-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px 14px 14px;
  margin-top: auto;
  border-top: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-surface-base) 35%, transparent);
}

.lv-scene-workshop__look-remove {
  margin-left: auto;
  color: var(--dq-label-tertiary);
}

.lv-scene-workshop__next {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-shrink: 0;
  padding: 14px 16px;
  margin-top: auto;
  border-radius: 12px;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--dq-accent) 16%, transparent) 0%,
    color-mix(in srgb, var(--dq-accent) 6%, transparent) 100%
  );
  border: 1px solid color-mix(in srgb, var(--dq-accent) 30%, var(--dq-border-subtle));
}

.lv-scene-workshop__next-kicker {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dq-accent);
  margin-bottom: 4px;
}

.lv-scene-workshop__next-text {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
}

@media (max-width: 900px) {
  .lv-scene-workshop__body {
    grid-template-columns: 1fr;
  }

  .lv-scene-workshop__sidebar {
    flex-direction: row;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .lv-scene-workshop__scene {
    width: auto;
    min-width: 168px;
  }

  .lv-scene-workshop__scene::before {
    display: none;
  }

  .lv-scene-workshop__detail-actions {
    width: 100%;
    padding-left: 0;
    border-left: none;
    padding-top: 8px;
    border-top: 0.5px solid var(--dq-border-subtle);
    justify-content: flex-end;
  }

  .lv-scene-workshop__looks {
    grid-template-columns: 1fr;
  }
}
</style>
