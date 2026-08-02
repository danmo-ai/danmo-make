<template>
  <section v-if="visibleRows.length || addableCharacters.length" class="lv-cast-looks">
    <div class="lv-cast-looks__label">{{ $tt('video.longVideoCastLooksTitle') }}</div>

    <div v-for="row in visibleRows" :key="row.characterId" class="lv-cast-looks__row">
      <span class="lv-cast-looks__name">{{ row.name }}</span>
      <DqSelect
        :id="rowSelectId(row.characterId)"
        :model-value="row.lookId"
        size="sm"
        variant="ghost"
        class="lv-cast-looks__select"
        @update:model-value="(v: string) => onLookChange(row.characterId, v)"
      >
        <DqOption
          v-for="opt in row.lookOptions"
          :key="opt.id"
          :label="opt.label"
          :value="opt.id"
        />
      </DqSelect>
      <DqIconButton
        type="text"
        size="sm"
        class="lv-cast-looks__remove"
        :label="$tt('common.delete')"
        @click="removeCharacter(row.characterId)"
      >
        <DqIcon :size="12"><Close /></DqIcon>
      </DqIconButton>
    </div>

    <DqSelect
      v-if="addableCharacters.length"
      :model-value="''"
      size="sm"
      variant="ghost"
      class="lv-cast-looks__add"
      :placeholder="$tt('video.longVideoCastLooksAdd')"
      @update:model-value="onAddCharacter"
    >
      <DqOption
        v-for="ch in addableCharacters"
        :key="ch.id"
        :label="ch.name"
        :value="ch.id"
      />
    </DqSelect>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { Close } from '@danqing/dq-shell';
import type { LongVideoCharacter, LongVideoShotCastLook } from '@/types';
import { charactersForShotCast, resolveShotCastLooks } from '@/utils/longVideoProject';

const props = defineProps<{
  characters: LongVideoCharacter[];
  castLooks: LongVideoShotCastLook[];
  sceneText: string;
  selectIdPrefix?: string;
}>();

const emit = defineEmits<(e: 'update:castLooks', value: LongVideoShotCastLook[]) => void>();

const resolvedCast = computed(() =>
  resolveShotCastLooks(props.characters, props.castLooks ?? [], props.sceneText),
);

const visibleRows = computed(() => {
  const castMap = new Map(resolvedCast.value.map((c) => [c.character_id, c.look_id]));
  const list = charactersForShotCast(props.characters, resolvedCast.value, props.sceneText);
  return list.map((ch) => ({
    characterId: ch.id,
    name: ch.name,
    lookId: castMap.get(ch.id) || ch.default_look_id || ch.looks[0]?.id || '',
    lookOptions: ch.looks.map((lk) => ({ id: lk.id, label: lk.label })),
  }));
});

const addableCharacters = computed(() => {
  const used = new Set(visibleRows.value.map((r) => r.characterId));
  return props.characters.filter((ch) => !used.has(ch.id));
});

function rowSelectId(characterId: string): string {
  return `${props.selectIdPrefix ?? 'lv-cast'}-${characterId}`;
}

function emitCast(next: LongVideoShotCastLook[]) {
  emit('update:castLooks', next);
}

function onLookChange(characterId: string, lookId: string) {
  const base = [...(props.castLooks?.length ? props.castLooks : resolvedCast.value)];
  const idx = base.findIndex((c) => c.character_id === characterId);
  if (idx >= 0) base[idx] = { character_id: characterId, look_id: lookId };
  else base.push({ character_id: characterId, look_id: lookId });
  emitCast(base);
}

function onAddCharacter(characterId: string) {
  if (!characterId) return;
  const ch = props.characters.find((c) => c.id === characterId);
  if (!ch) return;
  const lookId = ch.default_look_id || ch.looks[0]?.id || '';
  if (!lookId) return;
  const base = [...(props.castLooks?.length ? props.castLooks : resolvedCast.value)];
  if (base.some((c) => c.character_id === characterId)) return;
  base.push({ character_id: characterId, look_id: lookId });
  emitCast(base);
}

function removeCharacter(characterId: string) {
  const base = props.castLooks?.length
    ? [...props.castLooks]
    : [...resolvedCast.value];
  emitCast(base.filter((c) => c.character_id !== characterId));
}
</script>
