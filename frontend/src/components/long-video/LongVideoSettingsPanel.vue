<template>
  <section class="lv-settings" :class="{ 'lv-settings--inline': inline }">
    <template v-if="inline">
        <div class="lv-settings-bar" role="group" :aria-label="$tt('video.longVideoProjectSettings')">
          <div class="lv-settings-bar__field">
            <label class="lv-settings-bar__label" :for="ids.title">{{ $tt('video.longVideoProjectTitle') }}</label>
            <DqInput
              :id="ids.title"
              :model-value="title"
              size="sm"
              class="lv-settings-bar__control"
              :placeholder="$tt('video.longVideoProjectTitlePh')"
              @update:model-value="$emit('update:title', $event)"
            />
          </div>

          <div class="lv-settings-bar__field">
            <label class="lv-settings-bar__label" :for="ids.keyframe">{{ $tt('video.longVideoKeyframeModel') }}</label>
            <DqSelect
              :id="ids.keyframe"
              :model-value="keyframeModel"
              size="sm"
              variant="ghost"
              class="lv-settings-bar__control"
              :title="$tt('video.longVideoKeyframeModelHint')"
              @update:model-value="$emit('update:keyframeModel', $event)"
            >
              <DqOption v-for="optId in keyframeModelOptions" :key="optId" :label="modelLabel(optId)" :value="optId" />
            </DqSelect>
          </div>

          <div class="lv-settings-bar__field">
            <label class="lv-settings-bar__label" :for="ids.segment">{{ $tt('video.longVideoSegmentModel') }}</label>
            <DqSelect
              :id="ids.segment"
              :model-value="segmentModel"
              size="sm"
              variant="ghost"
              class="lv-settings-bar__control"
              :title="$tt('video.longVideoSegmentModelHint')"
              @update:model-value="$emit('update:segmentModel', $event)"
            >
              <DqOption v-for="optId in segmentModelOptions" :key="optId" :label="modelLabel(optId)" :value="optId" />
            </DqSelect>
          </div>

          <div class="lv-settings-bar__field">
            <label class="lv-settings-bar__label" :for="ids.size">{{ $tt('video.longVideoOutputSize') }}</label>
            <DqSelect
              :id="ids.size"
              :model-value="outputSize"
              size="sm"
              variant="ghost"
              class="lv-settings-bar__control"
              :title="$tt('video.longVideoOutputSizeHint')"
              @update:model-value="$emit('update:outputSize', $event)"
            >
              <DqOption
                v-for="opt in outputSizeOptions"
                :key="opt.value"
                :label="formatResolutionOptionLabel(opt)"
                :value="opt.value"
              />
            </DqSelect>
          </div>

          <div class="lv-settings-bar__field">
            <label class="lv-settings-bar__label" :for="ids.overlap">{{ $tt('video.longVideoOverlapFrames') }}</label>
            <DqSelect
              :id="ids.overlap"
              :model-value="overlapFrames"
              size="sm"
              variant="ghost"
              class="lv-settings-bar__control"
              :title="$tt('video.longVideoOverlapFramesHint')"
              @update:model-value="$emit('update:overlapFrames', Number($event))"
            >
              <DqOption
                v-for="n in overlapFrameOptions"
                :key="n"
                :label="overlapFrameLabel(n)"
                :value="n"
              />
            </DqSelect>
          </div>
        </div>
        <p v-if="inline && projectId" class="lv-settings-bar__project-id">
          <span>{{ $tt('video.longVideoProjectIdLabel') }}</span>
          <code :title="projectId">{{ shortProjectId }}</code>
          <DqButton size="sm" type="text" @click="copyProjectId">{{ $tt('studio.taskIdCopy') }}</DqButton>
        </p>
      </template>

      <template v-else>
        <DqPrefPane
          class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system lv-settings-pref-pane"
        >
          <DqPrefRow :label="$tt('video.longVideoProjectTitle')" stacked>
            <div class="settings-stacked-control">
              <DqInput
                :model-value="title"
                class="settings-mac-value-control"
                :placeholder="$tt('video.longVideoProjectTitlePh')"
                :aria-label="$tt('video.longVideoProjectTitle')"
                @update:model-value="$emit('update:title', $event)"
              />
            </div>
          </DqPrefRow>

        <DqPrefRow :label="$tt('video.longVideoKeyframeModel')" stacked>
          <div class="settings-stacked-control">
            <DqSelect
              :model-value="keyframeModel"
              class="settings-mac-value-control"
              :placeholder="$tt('video.longVideoKeyframeModel')"
              :aria-label="$tt('video.longVideoKeyframeModel')"
              @update:model-value="$emit('update:keyframeModel', $event)"
            >
              <DqOption v-for="optId in keyframeModelOptions" :key="optId" :label="modelLabel(optId)" :value="optId" />
            </DqSelect>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $tt('video.longVideoKeyframeModelHint') }}
            </p>
          </div>
        </DqPrefRow>

        <DqPrefRow :label="$tt('video.longVideoSegmentModel')" stacked>
          <div class="settings-stacked-control">
            <DqSelect
              :model-value="segmentModel"
              class="settings-mac-value-control"
              :placeholder="$tt('video.longVideoSegmentModel')"
              :aria-label="$tt('video.longVideoSegmentModel')"
              @update:model-value="$emit('update:segmentModel', $event)"
            >
              <DqOption v-for="optId in segmentModelOptions" :key="optId" :label="modelLabel(optId)" :value="optId" />
            </DqSelect>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $tt('video.longVideoSegmentModelHint') }}
            </p>
          </div>
        </DqPrefRow>

        <DqPrefRow :label="$tt('video.longVideoOutputSize')" stacked>
          <div class="settings-stacked-control">
            <DqSelect
              :model-value="outputSize"
              class="settings-mac-value-control"
              :placeholder="$tt('video.longVideoOutputSize')"
              :aria-label="$tt('video.longVideoOutputSize')"
              @update:model-value="$emit('update:outputSize', $event)"
            >
              <DqOption
                v-for="opt in outputSizeOptions"
                :key="opt.value"
                :label="formatResolutionOptionLabel(opt)"
                :value="opt.value"
              />
            </DqSelect>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $tt('video.longVideoOutputSizeHint') }}
            </p>
          </div>
        </DqPrefRow>

        <DqPrefRow :label="$tt('video.longVideoOverlapFrames')" stacked>
          <div class="settings-stacked-control">
            <DqSelect
              :model-value="overlapFrames"
              class="settings-mac-value-control"
              :placeholder="$tt('video.longVideoOverlapFrames')"
              :aria-label="$tt('video.longVideoOverlapFrames')"
              @update:model-value="$emit('update:overlapFrames', Number($event))"
            >
              <DqOption
                v-for="n in overlapFrameOptions"
                :key="n"
                :label="overlapFrameLabel(n)"
                :value="n"
              />
            </DqSelect>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $tt('video.longVideoOverlapFramesHint') }}
            </p>
          </div>
        </DqPrefRow>

        </DqPrefPane>
      </template>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import { shortActivityId } from '@/utils/longVideoActivity';
import { formatResolutionOptionLabel, type ResolutionSizeOption } from '@/utils/registryParamSchema';

const props = withDefaults(
  defineProps<{
    title: string;
    keyframeModel: string;
    segmentModel: string;
    outputSize: string;
    overlapFrames: number;
    outputSizeOptions: ResolutionSizeOption[];
    keyframeModelOptions: string[];
    segmentModelOptions: string[];
    modelLabel: (id: string) => string;
    inline?: boolean;
    projectId?: string;
  }>(),
  { inline: false },
);

defineEmits<{
  (e: 'update:title', value: string): void;
  (e: 'update:keyframeModel', value: string): void;
  (e: 'update:segmentModel', value: string): void;
  (e: 'update:outputSize', value: string): void;
  (e: 'update:overlapFrames', value: number): void;
}>();

const { t: $tt } = useI18n();

const shortProjectId = computed(() => shortActivityId(props.projectId ?? ''));

const overlapFrameOptions = [0, 4, 8, 16] as const;

const ids = {
  title: 'lv-settings-title',
  keyframe: 'lv-settings-keyframe',
  segment: 'lv-settings-segment',
  size: 'lv-settings-size',
  overlap: 'lv-settings-overlap',
} as const;

function overlapFrameLabel(n: number): string {
  return $tt('video.longVideoOverlapFramesOption', { n });
}

async function copyProjectId() {
  const id = (props.projectId ?? '').trim();
  if (!id) return;
  try {
    await navigator.clipboard.writeText(id);
    toast.success($tt('studio.taskIdCopied'));
  } catch {
    toast.error($tt('studio.error', { msg: 'clipboard' }));
  }
}
</script>
