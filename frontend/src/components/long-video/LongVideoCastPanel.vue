<template>
  <section class="lv-cast-workshop lv-section">
    <div v-if="scriptSourceLabel || characters.length" class="lv-cast-workshop__source">
      <span v-if="scriptSourceLabel" class="lv-cast-workshop__source-text">{{ scriptSourceLabel }}</span>
      <div class="lv-cast-workshop__source-actions">
        <div
          v-if="characters.length && totalLookCount > 0"
          class="lv-cast-workshop__progress"
          :title="$tt('video.longVideoCastReadyStat', { ready: readyPortraitCount, total: totalLookCount })"
        >
          <div class="lv-cast-workshop__progress-track">
            <div
              class="lv-cast-workshop__progress-fill"
              :style="{ width: `${Math.round((readyPortraitCount / totalLookCount) * 100)}%` }"
            />
          </div>
          <span class="lv-cast-workshop__stat">
            {{ $tt('video.longVideoCastReadyStat', { ready: readyPortraitCount, total: totalLookCount }) }}
          </span>
        </div>
        <DqButton
          v-if="missingPortraitCount > 0"
          size="sm"
          type="default"
          :loading="batchGenerating"
          :disabled="batchGenerating || portraitGeneratingKey != null"
          @click="emit('batch-generate-portraits')"
        >
          {{ $tt('video.longVideoCastPortraitBatch', { n: missingPortraitCount }) }}
        </DqButton>
        <DqButton v-if="canImport" type="text" size="sm" @click="importFromAnchor">
          {{ $tt('video.longVideoCastImportFromAnchor') }}
        </DqButton>
        <DqButton v-if="scriptSourceLabel" size="sm" type="text" @click="emit('go-to-script')">
          {{ $tt('video.longVideoCastEditScript') }}
        </DqButton>
      </div>
    </div>

    <div v-if="!characters.length" class="lv-cast-workshop__empty">
      <div class="lv-cast-workshop__empty-card">
        <p class="lv-cast-workshop__empty-text">{{ $tt('video.longVideoCastRosterEmpty') }}</p>
        <DqButton type="primary" @click="onAddCharacter">
          <DqIcon :size="12"><Plus /></DqIcon>
          {{ $tt('video.longVideoCastAddCharacter') }}
        </DqButton>
      </div>
    </div>

    <div v-else class="lv-cast-workshop__body">
      <nav class="lv-cast-workshop__sidebar" :aria-label="$tt('video.longVideoCastSidebarAria')">
        <button
          v-for="(ch, ci) in characters"
          :key="ch.id"
          type="button"
          class="lv-cast-workshop__char"
          :class="{ 'lv-cast-workshop__char--active': selectedIndex === ci }"
          @click="selectedIndex = ci"
        >
          <div
            class="lv-cast-workshop__char-thumb"
            :class="{ 'lv-cast-workshop__char-thumb--empty': !primaryPortraitAsset(ch) }"
          >
            <img v-if="primaryPortraitAsset(ch)" :src="portraitUrl(primaryPortraitAsset(ch)!)" alt="" />
            <span v-else class="lv-cast-workshop__char-initial">{{ characterInitial(ch.name) }}</span>
          </div>
          <div class="lv-cast-workshop__char-info">
            <span class="lv-cast-workshop__char-name">{{ ch.name.trim() || $tt('video.longVideoCastNewCharacterName') }}</span>
            <span class="lv-cast-workshop__char-meta">
              {{ $tt('video.longVideoCastLookCount', { n: ch.looks.length }) }}
              ·
              {{ $tt('video.longVideoCastCharReady', { n: portraitCountForCharacter(ch) }) }}
            </span>
          </div>
          <span
            v-if="!characterPortraitsComplete(ch)"
            class="lv-cast-workshop__char-warn"
            :title="$tt('video.longVideoCastPortraitMissing')"
          />
        </button>
        <button type="button" class="lv-cast-workshop__char lv-cast-workshop__char--add" @click="onAddCharacter">
          <span class="lv-cast-workshop__char-add-icon" aria-hidden="true">
            <DqIcon :size="14"><Plus /></DqIcon>
          </span>
          <span>{{ $tt('video.longVideoCastAddCharacter') }}</span>
        </button>
      </nav>

      <div v-if="selectedCharacter" class="lv-cast-workshop__detail">
        <header class="lv-cast-workshop__detail-head">
          <div class="lv-cast-workshop__detail-identity">
            <div
              class="lv-cast-workshop__detail-avatar"
              :class="{ 'lv-cast-workshop__detail-avatar--photo': Boolean(primaryPortraitAsset(selectedCharacter)) }"
            >
              <img
                v-if="primaryPortraitAsset(selectedCharacter)"
                :src="portraitUrl(primaryPortraitAsset(selectedCharacter)!)"
                alt=""
              />
              <span v-else>{{ characterInitial(selectedCharacter.name) }}</span>
            </div>
            <div class="lv-cast-workshop__detail-copy">
              <DqInput
                :model-value="selectedCharacter.name"
                size="sm"
                class="lv-cast-workshop__detail-name"
                :placeholder="$tt('video.longVideoCastNameExamplePh')"
                @update:model-value="onCharacterName(selectedIndex, $event)"
              />
              <div class="lv-cast-workshop__detail-meta">
                <span class="lv-cast-workshop__detail-chip">
                  {{ $tt('video.longVideoCastLookCount', { n: selectedCharacter.looks.length }) }}
                </span>
                <span
                  class="lv-cast-workshop__detail-chip"
                  :class="{ 'is-complete': characterPortraitsComplete(selectedCharacter) }"
                >
                  {{ $tt('video.longVideoCastCharReady', { n: portraitCountForCharacter(selectedCharacter) }) }}
                </span>
              </div>
            </div>
          </div>
          <div class="lv-cast-workshop__detail-actions">
            <DqButton type="primary" @click="addLook(selectedIndex)">
              <DqIcon :size="12"><Plus /></DqIcon>
              {{ $tt('video.longVideoCastAddLook') }}
            </DqButton>
            <DqIconButton
              type="text"
              size="sm"
              class="lv-cast-workshop__detail-delete"
              :label="$tt('video.longVideoCastRemove')"
              @click="removeCharacter(selectedIndex)"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
            </DqIconButton>
          </div>
        </header>

        <div class="lv-cast-workshop__looks">
          <article
            v-for="(lk, li) in selectedCharacter.looks"
            :key="lk.id"
            class="lv-cast-workshop__look-card"
            :class="{ 'lv-cast-workshop__look-card--ready': Boolean(lk.reference_asset_id) }"
          >
            <div
              class="lv-cast-workshop__look-portrait"
              :class="{ 'is-empty': !lk.reference_asset_id, 'is-loading': portraitGeneratingKey === portraitKey(selectedIndex, li) }"
            >
              <img v-if="lk.reference_asset_id" :src="portraitUrl(lk.reference_asset_id)" alt="" />
              <div v-else class="lv-cast-workshop__look-portrait-empty">
                <DqIcon :size="22"><PictureFilled /></DqIcon>
                <span>{{ $tt('video.longVideoCastPortraitMissing') }}</span>
              </div>
              <div class="lv-cast-workshop__look-badges">
                <span
                  class="lv-cast-workshop__look-badge"
                  :class="lk.reference_asset_id ? 'is-ready' : 'is-pending'"
                >
                  {{ lk.reference_asset_id ? $tt('video.longVideoCastPortraitReady') : $tt('video.longVideoCastPortraitMissing') }}
                </span>
                <span v-if="lk.lora_id" class="lv-cast-workshop__look-badge is-lora" :title="loraLabel(lk.lora_id)">
                  {{ loraLabel(lk.lora_id) }}
                </span>
              </div>
              <div
                v-if="portraitGeneratingKey === portraitKey(selectedIndex, li)"
                class="lv-cast-workshop__look-loading"
                aria-live="polite"
              >
                <DqIcon :size="20"><Loading /></DqIcon>
              </div>
              <div v-if="lk.reference_asset_id" class="lv-cast-workshop__look-portrait-overlay">
                <button
                  type="button"
                  class="lv-cast-workshop__look-overlay-btn"
                  :disabled="batchGenerating || portraitGeneratingKey != null"
                  @click="emit('pick-portrait-gallery', selectedIndex, li)"
                >
                  {{ $tt('video.longVideoCastPortraitImport') }}
                </button>
                <button
                  type="button"
                  class="lv-cast-workshop__look-overlay-btn lv-cast-workshop__look-overlay-btn--danger"
                  :disabled="batchGenerating || portraitGeneratingKey != null"
                  @click="emit('clear-portrait', selectedIndex, li)"
                >
                  {{ $tt('video.longVideoClearShort') }}
                </button>
              </div>
            </div>

            <div class="lv-cast-workshop__look-body">
              <label class="lv-cast-workshop__field">
                <span class="lv-cast-workshop__field-label">{{ $tt('video.longVideoCastLookLabelPh') }}</span>
                <DqInput
                  :model-value="lk.label"
                  size="sm"
                  class="lv-cast-workshop__look-label"
                  :placeholder="$tt('video.longVideoCastLookLabelExamplePh')"
                  @update:model-value="onLookLabel(selectedIndex, li, $event)"
                />
              </label>
              <label class="lv-cast-workshop__field">
                <span class="lv-cast-workshop__field-label">{{ $tt('video.longVideoCastLookBodyPh') }}</span>
                <DqInput
                  :model-value="lk.body"
                  type="textarea"
                  :rows="3"
                  size="sm"
                  class="lv-cast-workshop__look-desc"
                  :placeholder="$tt('video.longVideoCastLookBodyExamplePh')"
                  @update:model-value="onLookBody(selectedIndex, li, $event)"
                />
              </label>
              <label v-if="lk.reference_asset_id || lk.vision_description" class="lv-cast-workshop__field">
                <span class="lv-cast-workshop__field-label">{{ $tt('video.longVideoCastVisionDescriptionPh') }}</span>
                <DqInput
                  :model-value="lk.vision_description || ''"
                  type="textarea"
                  :rows="2"
                  size="sm"
                  class="lv-cast-workshop__look-vision"
                  :placeholder="$tt('video.longVideoCastVisionDescriptionExamplePh')"
                  @update:model-value="onLookVisionDescription(selectedIndex, li, $event)"
                />
              </label>
              <label v-if="compatibleLoras.length" class="lv-cast-workshop__field lv-cast-workshop__lora">
                <span class="lv-cast-workshop__field-label">{{ $tt('video.longVideoCastLoraLabel') }}</span>
                <DqSelect
                  :model-value="lk.lora_id || ''"
                  size="sm"
                  clearable
                  class="lv-cast-workshop__lora-select"
                  @update:model-value="onLookLora(selectedIndex, li, $event)"
                >
                  <DqOption :label="$tt('video.longVideoCastLoraNone')" value="" />
                  <DqOption
                    v-for="row in compatibleLoras"
                    :key="String(row.id || row.name)"
                    :label="String(row.name || row.id)"
                    :value="String(row.id || row.name)"
                  />
                </DqSelect>
              </label>
            </div>

            <footer class="lv-cast-workshop__look-foot">
              <DqButton
                size="sm"
                type="primary"
                :loading="portraitGeneratingKey === portraitKey(selectedIndex, li)"
                :disabled="batchGenerating || portraitGeneratingKey != null"
                @click="emit('generate-portrait', selectedIndex, li)"
              >
                {{ $tt('video.longVideoCastPortraitGenerate') }}
              </DqButton>
              <DqButton
                v-if="!lk.reference_asset_id"
                size="sm"
                type="default"
                :disabled="batchGenerating || portraitGeneratingKey != null"
                @click="emit('pick-portrait-gallery', selectedIndex, li)"
              >
                {{ $tt('video.longVideoCastPortraitImport') }}
              </DqButton>
              <DqButton
                v-if="lk.reference_asset_id"
                size="sm"
                type="text"
                :loading="visionBackfillKey === portraitKey(selectedIndex, li)"
                @click="emit('vision-backfill', selectedIndex, li)"
              >
                {{ $tt('video.longVideoCastVisionBackfill') }}
              </DqButton>
              <DqIconButton
                v-if="selectedCharacter.looks.length > 1"
                type="text"
                size="sm"
                class="lv-cast-workshop__look-remove"
                :label="$tt('common.delete')"
                @click="removeLook(selectedIndex, li)"
              >
                <DqIcon :size="12"><Close /></DqIcon>
              </DqIconButton>
            </footer>
            <LongVideoEntityRelatedTasks
              v-if="projectId"
              :project-id="projectId"
              phase="cast_portrait"
              :match="{ cast_character_id: selectedCharacter.id, cast_look_id: lk.id }"
              :label="$tt('video.longVideoRelatedTasksPortrait')"
              :refresh-token="portraitGeneratingKey"
            />
          </article>
        </div>
      </div>
    </div>

    <footer v-if="scriptParsed && characters.length" class="lv-cast-workshop__next">
      <div class="lv-cast-workshop__next-copy">
        <span class="lv-cast-workshop__next-kicker">{{ $tt('video.longVideoCastNextTitle') }}</span>
        <p class="lv-cast-workshop__next-text">{{ $tt('video.longVideoCastNextStoryboard') }}</p>
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
import type { LongVideoCharacter } from '@/types';
import {
  createCharacterEntry,
  createLookEntry,
  looksMissingPortrait,
  parseCharacterRosterFromAnchor,
} from '@/utils/longVideoProject';

const props = defineProps<{
  characters: LongVideoCharacter[];
  characterAnchor?: string;
  scriptSynopsis?: string;
  scriptParsed?: boolean;
  projectId?: string;
  compatibleLoras?: Record<string, unknown>[];
  portraitGeneratingKey?: string | null;
  visionBackfillKey?: string | null;
  batchGenerating?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:characters', value: LongVideoCharacter[]): void;
  (e: 'generate-portrait', characterIndex: number, lookIndex: number): void;
  (e: 'pick-portrait-gallery', characterIndex: number, lookIndex: number): void;
  (e: 'clear-portrait', characterIndex: number, lookIndex: number): void;
  (e: 'vision-backfill', characterIndex: number, lookIndex: number): void;
  (e: 'batch-generate-portraits'): void;
  (e: 'go-to-script'): void;
  (e: 'go-to-storyboard'): void;
  (e: 'import-style-anchor', value: string): void;
}>();

const { t: $tt, locale } = useI18n();

const selectedIndex = ref(0);

const compatibleLoras = computed(() => props.compatibleLoras ?? []);
const missingPortraitCount = computed(() => looksMissingPortrait(props.characters).length);
const totalLookCount = computed(() =>
  props.characters.reduce((sum, ch) => sum + ch.looks.length, 0),
);
const readyPortraitCount = computed(() =>
  props.characters.reduce(
    (sum, ch) => sum + ch.looks.filter((lk) => Boolean(lk.reference_asset_id)).length,
    0,
  ),
);

const selectedCharacter = computed(() => props.characters[selectedIndex.value] ?? null);

const scriptSourceLabel = computed(() => {
  if (!props.scriptParsed) return '';
  const n = props.characters.length;
  const looks = totalLookCount.value;
  if (n > 0) {
    return $tt('video.longVideoCastFromScript', { n, looks });
  }
  const syn = props.scriptSynopsis?.trim();
  if (syn) {
    const short = syn.length > 48 ? `${syn.slice(0, 48)}…` : syn;
    return $tt('video.longVideoCastFromScriptSynopsis', { text: short });
  }
  return $tt('video.longVideoCastFromScriptEmpty');
});

const canImport = computed(
  () => Boolean(props.characterAnchor?.trim()) && props.characters.length === 0,
);

watch(
  () => props.characters.length,
  (len) => {
    if (len === 0) {
      selectedIndex.value = 0;
      return;
    }
    if (selectedIndex.value >= len) selectedIndex.value = len - 1;
  },
);

function loc(): 'zh' | 'en' {
  return String(locale.value).startsWith('zh') ? 'zh' : 'en';
}

function portraitKey(ci: number, li: number): string {
  return `${ci}-${li}`;
}

function portraitUrl(assetId: string): string {
  return api.gallery.getImageUrl(`asset:${assetId}`);
}

function characterInitial(name: string): string {
  const trimmed = name.trim();
  return trimmed ? trimmed.slice(0, 1) : '?';
}

function primaryPortraitAsset(ch: LongVideoCharacter): string | undefined {
  for (const lk of ch.looks) {
    if (lk.reference_asset_id) return lk.reference_asset_id;
  }
  return undefined;
}

function portraitCountForCharacter(ch: LongVideoCharacter): number {
  return ch.looks.filter((lk) => Boolean(lk.reference_asset_id)).length;
}

function characterPortraitsComplete(ch: LongVideoCharacter): boolean {
  const ci = props.characters.indexOf(ch);
  if (ci < 0) return true;
  return !looksMissingPortrait(props.characters).some((m) => m.characterIndex === ci);
}

function loraLabel(loraId: string | undefined): string {
  if (!loraId) return '';
  const row = compatibleLoras.value.find((r) => String(r.id || r.name) === loraId);
  return String(row?.name || row?.id || loraId);
}

function emitCharacters(next: LongVideoCharacter[]) {
  emit('update:characters', next);
}

function onAddCharacter() {
  const next = [...props.characters, createCharacterEntry($tt('video.longVideoCastNewCharacterName'), loc())];
  emitCharacters(next);
  selectedIndex.value = next.length - 1;
}

function importFromAnchor() {
  const anchor = props.characterAnchor?.trim();
  if (!anchor) return;
  const parsed = parseCharacterRosterFromAnchor(anchor, loc());
  if (!parsed.characters.length) return;
  if (parsed.styleAnchor.trim()) {
    emit('import-style-anchor', parsed.styleAnchor);
  }
  emitCharacters(parsed.characters);
  selectedIndex.value = 0;
}

function onCharacterName(index: number, name: string) {
  emitCharacters(props.characters.map((c, i) => (i === index ? { ...c, name } : c)));
}

function removeCharacter(index: number) {
  const next = props.characters.filter((_, i) => i !== index);
  emitCharacters(next);
  selectedIndex.value = Math.min(selectedIndex.value, Math.max(0, next.length - 1));
}

function onLookLabel(ci: number, li: number, label: string) {
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci ? c : { ...c, looks: c.looks.map((lk, j) => (j === li ? { ...lk, label } : lk)) },
    ),
  );
}

function onLookBody(ci: number, li: number, body: string) {
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci ? c : { ...c, looks: c.looks.map((lk, j) => (j === li ? { ...lk, body } : lk)) },
    ),
  );
}

function onLookVisionDescription(ci: number, li: number, visionDescription: string) {
  const trimmed = visionDescription.trim();
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci
        ? c
        : {
            ...c,
            looks: c.looks.map((lk, j) => {
              if (j !== li) return lk;
              const next = { ...lk, vision_description: trimmed || undefined };
              if (!trimmed) delete next.vision_description;
              return next;
            }),
          },
    ),
  );
}

function onLookLora(ci: number, li: number, loraId: string | number | boolean) {
  const id = String(loraId || '').trim() || undefined;
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci
        ? c
        : { ...c, looks: c.looks.map((lk, j) => (j === li ? { ...lk, lora_id: id } : lk)) },
    ),
  );
}

function addLook(ci: number) {
  const ch = props.characters[ci];
  if (!ch) return;
  emitCharacters(
    props.characters.map((c, i) =>
      i === ci
        ? { ...c, looks: [...c.looks, createLookEntry(c.name, $tt('video.longVideoCastNewLookLabel'), loc())] }
        : c,
    ),
  );
}

function removeLook(ci: number, li: number) {
  emitCharacters(
    props.characters.map((c, i) => {
      if (i !== ci) return c;
      const looks = c.looks.filter((_, j) => j !== li);
      const default_look_id =
        c.default_look_id === c.looks[li]?.id ? looks[0]?.id ?? '' : c.default_look_id;
      return { ...c, looks, default_look_id };
    }),
  );
}
</script>

<style scoped>
.lv-cast-workshop {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 16px 16px;
  width: 100%;
  box-sizing: border-box;
}

.lv-cast-workshop__source {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  padding-bottom: 10px;
  border-bottom: 0.5px solid var(--dq-border-subtle);
}

.lv-cast-workshop__source-text {
  flex: 1;
  min-width: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.lv-cast-workshop__source-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.lv-cast-workshop__progress {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 140px;
}

.lv-cast-workshop__progress-track {
  width: 72px;
  height: 5px;
  border-radius: 999px;
  overflow: hidden;
  background: color-mix(in srgb, var(--dq-fill-control) 70%, transparent);
}

.lv-cast-workshop__progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--dq-accent) 85%, white),
    var(--dq-accent)
  );
  transition: width 0.25s ease;
}

.lv-cast-workshop__stat {
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

.lv-cast-workshop__empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 240px;
}

.lv-cast-workshop__empty-card {
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

.lv-cast-workshop__empty-text {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  text-align: center;
  max-width: 280px;
}

.lv-cast-workshop__body {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 12px;
  overflow: hidden;
}

.lv-cast-workshop__sidebar {
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

.lv-cast-workshop__char {
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

.lv-cast-workshop__char::before {
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

.lv-cast-workshop__char:hover {
  background: color-mix(in srgb, var(--dq-fill-control) 65%, transparent);
}

.lv-cast-workshop__char--active {
  background: color-mix(in srgb, var(--dq-accent) 11%, transparent);
  border-color: color-mix(in srgb, var(--dq-accent) 32%, transparent);
  box-shadow: 0 2px 10px color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lv-cast-workshop__char--active::before {
  background: var(--dq-accent);
}

.lv-cast-workshop__char--add {
  justify-content: center;
  padding-left: 10px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  border-style: dashed;
  border-color: var(--dq-border-subtle);
  margin-top: 4px;
}

.lv-cast-workshop__char--add::before {
  display: none;
}

.lv-cast-workshop__char-thumb {
  width: 42px;
  height: 52px;
  flex-shrink: 0;
  border-radius: 9px;
  overflow: hidden;
  background: color-mix(in srgb, var(--dq-bg-base) 80%, #000);
  border: 0.5px solid color-mix(in srgb, white 8%, var(--dq-border-subtle));
  box-shadow: 0 2px 8px color-mix(in srgb, black 25%, transparent);
}

.lv-cast-workshop__char-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-cast-workshop__char-thumb--empty {
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

.lv-cast-workshop__char-initial {
  font-size: var(--dq-font-size-title);
  font-weight: 700;
  color: var(--dq-label-secondary);
}

.lv-cast-workshop__char-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.lv-cast-workshop__char-name {
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  color: var(--dq-label-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.lv-cast-workshop__char-meta {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-tertiary);
}

.lv-cast-workshop__char-warn {
  width: 8px;
  height: 8px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--dq-warning);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-warning) 28%, transparent);
}

.lv-cast-workshop__char-add-icon {
  display: flex;
}

.lv-cast-workshop__detail {
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

.lv-cast-workshop__detail-head {
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

.lv-cast-workshop__detail-identity {
  display: flex;
  align-items: center;
  gap: 14px;
  flex: 1;
  min-width: 0;
}

.lv-cast-workshop__detail-avatar {
  width: 50px;
  height: 60px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 11px;
  overflow: hidden;
  font-size: var(--dq-font-size-body);
  font-weight: 700;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 32%, transparent);
  box-shadow:
    0 0 0 2px color-mix(in srgb, var(--dq-accent) 12%, transparent),
    0 4px 14px color-mix(in srgb, black 22%, transparent);
}

.lv-cast-workshop__detail-avatar--photo {
  padding: 0;
  background: var(--dq-color-black);
  border-color: color-mix(in srgb, white 10%, var(--dq-border-subtle));
}

.lv-cast-workshop__detail-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-cast-workshop__detail-copy {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-cast-workshop__detail-name {
  width: 100%;
  max-width: 420px;
}

.lv-cast-workshop__detail-name :deep(.dq-input) {
  width: 100%;
  padding: 0;
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  letter-spacing: var(--dq-tracking-tighter);
  line-height: 1.25;
  color: var(--dq-label-primary);
  background: transparent;
  border: none;
  box-shadow: none;
  border-radius: 8px;
  transition: background 0.15s, box-shadow 0.15s;
}

.lv-cast-workshop__detail-name :deep(.dq-input::placeholder) {
  color: var(--dq-label-tertiary);
  font-weight: 500;
}

.lv-cast-workshop__detail-name :deep(.dq-input:hover:not(:disabled)) {
  background: color-mix(in srgb, var(--dq-fill-control) 35%, transparent);
}

.lv-cast-workshop__detail-name :deep(.dq-input:focus) {
  outline: none;
  background: color-mix(in srgb, var(--dq-fill-control) 50%, transparent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 22%, transparent);
  padding: 4px 10px;
}

.lv-cast-workshop__detail-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.lv-cast-workshop__detail-chip {
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

.lv-cast-workshop__detail-chip.is-complete {
  color: color-mix(in srgb, var(--dq-success) 85%, white);
  background: color-mix(in srgb, var(--dq-success) 12%, transparent);
  border-color: color-mix(in srgb, var(--dq-success) 35%, transparent);
}

.lv-cast-workshop__detail-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  padding-left: 12px;
  border-left: 0.5px solid color-mix(in srgb, var(--dq-border-subtle) 80%, transparent);
}

.lv-cast-workshop__detail-delete {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  color: var(--dq-label-tertiary);
  background: color-mix(in srgb, var(--dq-fill-control) 40%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-cast-workshop__detail-delete:hover {
  color: var(--dq-danger);
  background: color-mix(in srgb, var(--dq-danger) 10%, transparent);
  border-color: color-mix(in srgb, var(--dq-danger) 35%, transparent);
}

.lv-cast-workshop__danger {
  color: var(--dq-label-tertiary);
}

.lv-cast-workshop__danger:hover {
  color: var(--dq-danger);
}

.lv-cast-workshop__looks {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
  align-content: start;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.lv-cast-workshop__look-card {
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

.lv-cast-workshop__look-card:hover {
  transform: translateY(-2px);
  border-color: color-mix(in srgb, var(--dq-accent) 28%, var(--dq-border-subtle));
  box-shadow:
    0 1px 0 color-mix(in srgb, white 5%, transparent),
    0 12px 28px color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.lv-cast-workshop__look-card--ready {
  border-color: color-mix(in srgb, var(--dq-success) 22%, var(--dq-border-subtle));
}

.lv-cast-workshop__look-portrait {
  position: relative;
  aspect-ratio: 9 / 16;
  max-height: 360px;
  overflow: hidden;
  background: #080808;
  border-bottom: 0.5px solid var(--dq-border-subtle);
}

.lv-cast-workshop__look-portrait img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-cast-workshop__look-badges {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  pointer-events: none;
}

.lv-cast-workshop__look-badge {
  padding: 3px 8px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: 0.01em;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border: 0.5px solid rgba(255, 255, 255, 0.12);
}

.lv-cast-workshop__look-badge.is-pending {
  color: #ffd89a;
  background: color-mix(in srgb, var(--dq-warning) 55%, rgba(0, 0, 0, 0.5));
}

.lv-cast-workshop__look-badge.is-ready {
  color: #b8f5c8;
  background: color-mix(in srgb, var(--dq-success) 45%, rgba(0, 0, 0, 0.5));
}

.lv-cast-workshop__look-badge.is-lora {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #b8d4ff;
  background: color-mix(in srgb, var(--dq-accent) 50%, rgba(0, 0, 0, 0.5));
}

.lv-cast-workshop__look-portrait-empty {
  height: 100%;
  min-height: 200px;
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

.lv-cast-workshop__look-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, black 50%, transparent);
  color: white;
}

.lv-cast-workshop__look-portrait-overlay {
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

.lv-cast-workshop__look-portrait:hover .lv-cast-workshop__look-portrait-overlay {
  opacity: 1;
}

.lv-cast-workshop__look-overlay-btn {
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

.lv-cast-workshop__look-overlay-btn:hover {
  background: rgba(255, 255, 255, 0.18);
}

.lv-cast-workshop__look-overlay-btn--danger {
  border-color: color-mix(in srgb, var(--dq-danger) 50%, transparent);
  color: #ffc9c9;
}

.lv-cast-workshop__look-body {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 14px 14px 0;
}

.lv-cast-workshop__field {
  display: flex;
  flex-direction: column;
  gap: 5px;
  margin: 0;
}

.lv-cast-workshop__field-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: var(--dq-tracking-wide);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.lv-cast-workshop__look-desc :deep(textarea) {
  min-height: 76px;
  resize: vertical;
}

.lv-cast-workshop__lora-select {
  width: 100%;
}

.lv-cast-workshop__look-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 12px 14px 14px;
  margin-top: auto;
  border-top: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-bg-base) 35%, transparent);
}

.lv-cast-workshop__look-remove {
  margin-left: auto;
  color: var(--dq-label-tertiary);
}

.lv-cast-workshop__next {
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

.lv-cast-workshop__next-kicker {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-widest);
  text-transform: uppercase;
  color: var(--dq-accent);
  margin-bottom: 4px;
}

.lv-cast-workshop__next-text {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
}

@media (max-width: 900px) {
  .lv-cast-workshop__body {
    grid-template-columns: 1fr;
  }

  .lv-cast-workshop__sidebar {
    flex-direction: row;
    overflow-x: auto;
    overflow-y: hidden;
  }

  .lv-cast-workshop__char {
    width: auto;
    min-width: 168px;
  }

  .lv-cast-workshop__char::before {
    display: none;
  }

  .lv-cast-workshop__detail-actions {
    width: 100%;
    padding-left: 0;
    border-left: none;
    padding-top: 8px;
    border-top: 0.5px solid var(--dq-border-subtle);
    justify-content: flex-end;
  }

  .lv-cast-workshop__looks {
    grid-template-columns: 1fr;
  }
}
</style>
