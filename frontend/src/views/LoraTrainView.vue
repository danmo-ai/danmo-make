<template>
  <div class="copilot-page lora-train-page">
    <div class="copilot-page__sidebar">
      <DqSurfaceCard class="lv-sidebar-card lora-train-page__sidebar-card">
        <div class="card-title">
          <DqIcon><MagicStick /></DqIcon>
          {{ $t('loraTrain.recentRuns') }}
        </div>
        <DqButton
          type="primary"
          block
          class="lv-sidebar-card__new-btn"
          @click="startNewTraining"
        >
          + {{ $t('loraTrain.newTraining') }}
        </DqButton>
        <div class="lora-train-page__sidebar-body">
          <LoraTrainHistory
            ref="historyRef"
            :active-id="activeRunId"
            variant="sidebar"
            hide-header
            hide-refresh
            @select="openRun"
          />
        </div>
      </DqSurfaceCard>
    </div>

    <div class="copilot-page__main">
      <LoraTrainRunDetail
        v-if="activeRunId"
        :task-id="activeRunId"
        @back="onRunBack"
        @verify="onVerifyGenerate"
      />

      <template v-else>
        <div class="lv-editor-tabs-wrapper">
          <div class="lv-editor-tabs lora-train-page__stepper" role="tablist">
            <template v-for="(item, i) in stepNavItems" :key="item.key">
              <span v-if="i > 0" class="lv-editor-tabs__connector" aria-hidden="true" />
              <button
                type="button"
                role="tab"
                class="lv-editor-tabs__tab"
                :class="{
                  'lv-editor-tabs__tab--active': step === i,
                  'lv-editor-tabs__tab--done': step > i || stepDone(i),
                  'lv-editor-tabs__tab--locked': i > maxReachableStep,
                }"
                :aria-selected="step === i"
                @click="goToStep(i)"
              >
                <span class="lv-editor-tabs__icon" :class="{ 'lv-editor-tabs__icon--done': step > i || stepDone(i) }" aria-hidden="true">
                  <span v-if="step > i || stepDone(i)" class="lv-editor-tabs__icon-mark">✓</span>
                  <span v-else class="lv-editor-tabs__icon-mark">{{ i + 1 }}</span>
                </span>
                <span class="lv-editor-tabs__text">
                  <span class="lv-editor-tabs__label">{{ item.label }}</span>
                </span>
              </button>
            </template>
          </div>
        </div>

        <DqSurfaceCard class="copilot-page__workspace-card studio-surface-card">
          <!-- Step 1: base model -->
          <div v-show="step === 0" class="lora-train-page__panel lora-train-page__panel--base">
            <div class="lora-train-page__base-grid">
              <button
                v-for="m in trainableModels"
                :key="m.id"
                type="button"
                class="lora-train-page__base-tile"
                :class="{
                  'is-selected': form.base_model === m.id,
                  'is-ready': m.trainable && m.ready,
                  [`is-family-${modelFamilyKey(m.id)}`]: true,
                }"
                :disabled="!m.trainable || !m.ready"
                :aria-pressed="form.base_model === m.id"
                @click="selectBase(m)"
              >
                <span class="lora-train-page__base-tile-icon" aria-hidden="true">
                  {{ modelInitials(m) }}
                </span>
                <span class="lora-train-page__base-tile-body">
                  <span class="lora-train-page__base-tile-name">{{ modelDisplayName(m) }}</span>
                  <span v-if="modelShortDesc(m)" class="lora-train-page__base-tile-desc">
                    {{ modelShortDesc(m) }}
                  </span>
                </span>
                <span
                  v-if="form.base_model === m.id"
                  class="lora-train-page__base-tile-check"
                  aria-hidden="true"
                />
              </button>
            </div>
          </div>
          <div v-show="step === 0" class="lora-train-page__nav-bar">
            <span />
            <DqButton
              type="primary"
              :disabled="!canNext(0)"
              @click="goToStep(1)"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 2: dataset -->
          <div v-show="step === 1" class="lora-train-page__panel lora-train-page__panel--flush">
            <LoraDatasetPanel
              ref="datasetPanelRef"
              :selected-id="form.dataset_id"
              :datasets="datasets"
              :default-prompt="form.default_prompt"
              :caption-edits="captionEdits"
              :dataset-health="datasetHealth"
              :vision-available="visionAvailable"
              :dataset-vlm-loading="datasetVlmLoading"
              @update:selected-id="form.dataset_id = $event"
              @update:default-prompt="form.default_prompt = $event"
              @update:caption-edits="onCaptionEditsUpdate"
              @datasets-changed="onDatasetsChanged"
              @run-dataset-vlm="runDatasetVlmAudit"
            />
          </div>
          <div v-show="step === 1" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(0)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :disabled="!canNext(1)"
              @click="goToStep(2)"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 3: config -->
          <div v-show="step === 2" class="lora-train-page__panel">
            <DqAlert
              v-if="hasResume"
              type="warning"
              :closable="false"
              class="lora-train-page__resume-banner"
              :title="$t('loraTrain.resumeActiveTitle')"
            >
              <p>{{ $t('loraTrain.resumeActiveBody', {
                task: form.resume_task_id,
                checkpoint: form.resume_checkpoint,
                base: form.base_model,
              }) }}</p>
              <DqButton size="sm" type="secondary" @click="clearResume">
                {{ $t('loraTrain.clearResume') }}
              </DqButton>
            </DqAlert>
            <div class="lora-train-page__preset-grid">
              <button
                v-for="p in presetKeys"
                :key="p"
                type="button"
                class="lora-train-page__preset-card"
                :class="{
                  'is-selected': form.preset === p,
                  'is-recommended': p === 'scheme4' && form.base_model === 'z-image',
                }"
                @click="form.preset = p"
              >
                <span class="lora-train-page__preset-name">
                  {{ $t(`loraTrain.preset.${p}`) }}
                  <em
                    v-if="p === 'scheme4' && form.base_model === 'z-image'"
                    class="lora-train-page__preset-badge"
                  >{{ $t('loraTrain.presetRecommended') }}</em>
                </span>
                <span class="lora-train-page__preset-desc">{{ $t(`loraTrain.presetDesc.${p}`) }}</span>
                <span v-if="p !== 'custom' && presetStats(p)" class="lora-train-page__preset-stats">
                  {{ presetStats(p) }}
                </span>
              </button>
            </div>
            <DqAlert
              v-if="form.base_model === 'z-image' && form.preset === 'scheme4'"
              type="info"
              :closable="false"
              class="lora-train-page__scheme4-hint"
              :title="$t('loraTrain.preset.scheme4')"
            >
              <p>{{ $t('loraTrain.scheme4InferHint') }}</p>
            </DqAlert>
            <div v-if="activePresetDetail" class="lora-train-page__preset-detail">
              <span class="lora-train-page__preset-detail-label">{{ $t('loraTrain.presetDetail') }}</span>
              <div class="lora-train-page__preset-detail-grid">
                <span>{{ $t('loraTrain.iterations') }}: {{ activePresetDetail.iterations }}</span>
                <span>{{ $t('loraTrain.loraRank') }}: {{ activePresetDetail.lora_rank }}</span>
                <span v-if="presetResolution(activePresetDetail)">
                  {{ $t('loraTrain.resolution') }}: {{ presetResolution(activePresetDetail) }}
                  <em class="lora-train-page__preset-crop-hint">({{ $t('loraTrain.resolutionAutoCrop') }})</em>
                </span>
                <span v-if="activePresetDetail.learning_rate">
                  {{ $t('loraTrain.learningRate') }}: {{ activePresetDetail.learning_rate }}
                </span>
                <span v-if="activePresetDetail.optimizer">
                  {{ $t('loraTrain.optimizer') }}: {{ activePresetDetail.optimizer }}
                </span>
                <span v-if="activePresetDetail.grad_checkpoint">
                  {{ $t('loraTrain.gradCheckpoint') }}
                </span>
                <span v-if="activePresetDetail.val_split">
                  {{ $t('loraTrain.summaryValSplit', { pct: Math.round(Number(activePresetDetail.val_split) * 100) }) }}
                </span>
              </div>
            </div>

            <div v-if="form.preset === 'custom'" class="lora-train-page__custom-block">
              <DqAlert type="info" :closable="false" :title="$t('loraTrain.customGuideTitle')">
                <p class="lora-train-page__custom-guide-text">{{ $t('loraTrain.customGuideBody') }}</p>
                <ul class="lora-train-page__custom-guide-list">
                  <li>{{ $t('loraTrain.customGuideTip1') }}</li>
                  <li>{{ $t('loraTrain.customGuideTip2') }}</li>
                  <li>{{ $t('loraTrain.customGuideTip3') }}</li>
                  <li>{{ $t('loraTrain.customGuideTip4') }}</li>
                </ul>
              </DqAlert>
              <div class="lora-train-page__field-grid">
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.iterations') }}</label>
                  <DqInput v-model.number="form.iterations" type="number" min="50" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customIterationsHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.loraRank') }}</label>
                  <DqInput v-model.number="form.lora_rank" type="number" min="1" max="64" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customRankHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.learningRate') }}</label>
                  <DqInput v-model.number="form.learning_rate" type="number" step="0.00001" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.resolutionEdge') }}</label>
                  <DqInput v-model.number="form.resolution_edge" type="number" min="256" max="1024" step="8" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customResolutionHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.progressSteps') }}</label>
                  <DqInput v-model.number="form.progress_steps" type="number" min="4" max="100" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.progressStepsHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.minSnrGamma') }}</label>
                  <DqInput v-model.number="form.min_snr_gamma" type="number" step="1" min="0" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customMinSnrHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.loraBlocks') }}</label>
                  <DqInput v-model.number="form.lora_blocks" type="number" min="1" max="64" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customBlocksHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.numAugmentations') }}</label>
                  <DqInput v-model.number="form.num_augmentations" type="number" min="1" max="10" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customAugmentHint') }}</p>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.gradAccumulate') }}</label>
                  <DqInput v-model.number="form.grad_accumulate" type="number" min="1" max="32" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customGradAccumHint') }}</p>
                </div>
                <div v-if="supportsPriorPreservation" class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.priorLossWeight') }}</label>
                  <DqInput v-model.number="form.prior_loss_weight" type="number" step="0.1" min="0" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customPriorHint') }}</p>
                </div>
                <div v-if="supportsPriorPreservation" class="lora-train-page__field lora-train-page__field--wide">
                  <label class="lora-train-page__label">{{ $t('loraTrain.classPrompt') }}</label>
                  <DqInput
                    v-model="form.class_prompt"
                    :placeholder="$t('loraTrain.classPromptPlaceholder')"
                  />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.customClassPromptHint') }}</p>
                </div>
              </div>
            </div>

            <div class="lora-train-page__optimization">
              <div class="lora-train-page__optimization-head">
                <span class="lora-train-page__label">{{ $t('loraTrain.optimization') }}</span>
                <p class="lora-train-page__field-hint">{{ $t('loraTrain.optimizationDesc') }}</p>
              </div>
              <div class="lora-train-page__field-grid">
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.qloraBits') }}</label>
                  <DqSelect v-model="form.qlora_bits">
                    <DqOption :label="$t('loraTrain.qloraOff')" value="" />
                    <DqOption :label="$t('loraTrain.qlora4')" value="4" />
                    <DqOption :label="$t('loraTrain.qlora8')" value="8" />
                  </DqSelect>
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.qloraHint') }}</p>
                </div>
                <div class="lora-train-page__field lora-train-page__field--switch">
                  <label class="lora-train-page__label">{{ $t('loraTrain.gradCheckpoint') }}</label>
                  <DqSwitch v-model="form.grad_checkpoint" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.gradCheckpointHint') }}</p>
                </div>
                <div class="lora-train-page__field lora-train-page__field--switch">
                  <label class="lora-train-page__label">{{ $t('loraTrain.compileStep') }}</label>
                  <DqSwitch v-model="form.compile_step" />
                  <p class="lora-train-page__field-hint">{{ $t('loraTrain.compileStepHint') }}</p>
                </div>
              </div>
            </div>

            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.progressPrompt') }}</label>
              <DqInput v-model="form.progress_prompt" type="textarea" :rows="3" />
              <p class="lora-train-page__field-hint">{{ $t('loraTrain.progressPromptDesc') }}</p>
              <DqAlert
                v-if="promptCaptionMismatch"
                type="warning"
                :closable="false"
                class="lora-train-page__prompt-warn"
                :title="$t('loraTrain.promptMismatchTitle')"
              >
                <p>{{ $t('loraTrain.promptMismatchBody', { dataset: effectiveDatasetCaption, progress: form.progress_prompt.trim() }) }}</p>
                <DqButton size="sm" type="secondary" @click="syncProgressPromptFromDataset">
                  {{ $t('loraTrain.promptMismatchSync') }}
                </DqButton>
              </DqAlert>
            </div>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.outputName') }}</label>
              <DqInput v-model="form.output_name" :placeholder="$t('loraTrain.outputNamePlaceholder')" />
              <p class="lora-train-page__field-hint">{{ $t('loraTrain.outputNameDesc') }}</p>
            </div>
            <details class="lora-train-page__advanced">
              <summary>{{ form.preset === 'custom' ? $t('loraTrain.advanced') : $t('loraTrain.expertTuning') }}</summary>
              <div class="lora-train-page__field-grid">
                <div v-if="form.preset !== 'custom'" class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.loraBlocks') }}</label>
                  <DqInput v-model.number="form.lora_blocks" type="number" :placeholder="$t('loraTrain.usePresetDefault')" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.loraAlpha') }}</label>
                  <DqInput v-model.number="form.lora_alpha" type="number" step="1" min="1" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.loraDropout') }}</label>
                  <DqInput v-model.number="form.lora_dropout" type="number" step="0.05" min="0" max="0.5" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.optimizer') }}</label>
                  <DqSelect v-model="form.optimizer">
                    <DqOption :label="$t('loraTrain.usePresetDefault')" value="" />
                    <DqOption :label="$t('loraTrain.optimizerAdam')" value="adam" />
                    <DqOption :label="$t('loraTrain.optimizerAdamw')" value="adamw" />
                  </DqSelect>
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.weightDecay') }}</label>
                  <DqInput v-model.number="form.weight_decay" type="number" step="0.01" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.valSplit') }}</label>
                  <DqInput v-model.number="form.val_split" type="number" step="0.05" min="0" max="0.5" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.valEvery') }}</label>
                  <DqInput v-model.number="form.val_every" type="number" />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.trainType') }}</label>
                  <DqSelect v-model="form.train_type">
                    <DqOption :label="$t('loraTrain.usePresetDefault')" value="" />
                    <DqOption label="LoRA" value="lora" />
                    <DqOption label="DoRA" value="dora" />
                  </DqSelect>
                </div>
                <div v-if="form.preset !== 'custom'" class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.minSnrGamma') }}</label>
                  <DqInput v-model.number="form.min_snr_gamma" type="number" step="1" min="0" />
                </div>
                <div v-if="supportsPriorPreservation && form.preset !== 'custom'" class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.priorLossWeight') }}</label>
                  <DqInput v-model.number="form.prior_loss_weight" type="number" step="0.1" min="0" />
                </div>
                <div v-if="supportsPriorPreservation && form.preset !== 'custom'" class="lora-train-page__field lora-train-page__field--wide">
                  <label class="lora-train-page__label">{{ $t('loraTrain.classPrompt') }}</label>
                  <DqInput
                    v-model="form.class_prompt"
                    :placeholder="$t('loraTrain.classPromptPlaceholder')"
                  />
                </div>
                <div class="lora-train-page__field">
                  <label class="lora-train-page__label">{{ $t('loraTrain.earlyStopPatience') }}</label>
                  <DqInput v-model.number="form.early_stop_patience" type="number" min="0" />
                </div>
                <div class="lora-train-page__field lora-train-page__field--switch">
                  <label class="lora-train-page__label">{{ $t('loraTrain.fuseAdapters') }}</label>
                  <DqSwitch v-model="form.fuse_adapters" />
                </div>
              </div>
              <div class="lora-train-page__field">
                <label class="lora-train-page__label">{{ $t('loraTrain.loraModuleKeys') }}</label>
                <DqInput
                  v-model="form.lora_module_keys"
                  :placeholder="$t('loraTrain.loraModuleKeysPlaceholder')"
                />
              </div>
            </details>
          </div>
          <div v-show="step === 2" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(1)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :disabled="!canNext(2)"
              @click="goToStep(3)"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 4: confirm -->
          <div v-show="step === 3" class="lora-train-page__panel">
            <DqAlert
              v-if="hasResume"
              type="warning"
              :closable="false"
              class="lora-train-page__resume-banner"
              :title="$t('loraTrain.resumeConfirmWarning')"
            >
              <p>{{ $t('loraTrain.resumeActiveBody', {
                task: form.resume_task_id,
                checkpoint: form.resume_checkpoint,
                base: form.base_model,
              }) }}</p>
              <DqButton size="sm" type="secondary" @click="clearResume">
                {{ $t('loraTrain.clearResume') }}
              </DqButton>
            </DqAlert>
            <p class="lora-train-page__confirm-intro">{{ $t('loraTrain.confirmIntro') }}</p>
            <LoraQualityHints
              v-if="datasetHealth"
              :report="datasetHealth"
              mode="dataset"
              :vision-available="visionAvailable"
              :vlm-loading="datasetVlmLoading"
              :dataset-audit-kind="selectedDataset?.kind === 'style' ? 'style' : 'concept'"
              class="lora-train-page__confirm-health"
              @run-vlm="runDatasetVlmAudit"
            />
            <div v-if="confirmPreviewImages.length" class="lora-train-page__confirm-preview">
              <span class="lora-train-page__confirm-preview-label">{{ $t('loraTrain.confirmPreview') }}</span>
              <div class="lora-train-page__confirm-preview-grid">
                <img
                  v-for="img in confirmPreviewImages"
                  :key="img.file"
                  :src="confirmImageUrl(img.file)"
                  :alt="img.file"
                  loading="lazy"
                />
                <span
                  v-if="(selectedDataset?.image_count || 0) > confirmPreviewImages.length"
                  class="lora-train-page__confirm-preview-more"
                >
                  +{{ (selectedDataset?.image_count || 0) - confirmPreviewImages.length }}
                </span>
              </div>
            </div>
            <DqAlert
              v-if="form.base_model === 'z-image' && form.preset === 'scheme4'"
              type="info"
              :closable="false"
              class="lora-train-page__scheme4-hint"
            >
              <p>{{ $t('loraTrain.scheme4InferHint') }}</p>
            </DqAlert>
            <ul class="lora-train-page__summary">
              <li><span>{{ $t('loraTrain.summaryBase') }}</span><strong>{{ form.base_model }}</strong></li>
              <li>
                <span>{{ $t('loraTrain.summaryDataset') }}</span>
                <strong>{{ selectedDataset?.name }} ({{ selectedDataset?.image_count }})</strong>
              </li>
              <li><span>{{ $t('loraTrain.summaryPreset') }}</span><strong>{{ $t(`loraTrain.preset.${form.preset}`) }}</strong></li>
              <li><span>{{ $t('loraTrain.summaryPrompt') }}</span><strong>{{ form.progress_prompt }}</strong></li>
              <li v-if="effectiveDatasetCaption && effectiveDatasetCaption !== form.progress_prompt.trim()">
                <span>{{ $t('loraTrain.summaryDatasetCaption') }}</span>
                <strong>{{ effectiveDatasetCaption }}</strong>
              </li>
              <li v-if="form.output_name.trim()">
                <span>{{ $t('loraTrain.summaryOutput') }}</span><strong>{{ form.output_name }}</strong>
              </li>
              <li v-if="confirmOptimizationSummary">
                <span>{{ $t('loraTrain.summaryOptimization') }}</span>
                <strong>{{ confirmOptimizationSummary }}</strong>
              </li>
              <li v-if="hasResume">
                <span>{{ $t('loraTrain.summaryResume') }}</span>
                <strong>{{ form.resume_task_id }} / {{ form.resume_checkpoint }}</strong>
              </li>
            </ul>
          </div>
          <div v-show="step === 3" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(2)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :loading="submitting"
              :disabled="!canSubmit"
              @click="submitTraining"
            >
              {{ $t('loraTrain.startTraining') }}
            </DqButton>
          </div>
        </DqSurfaceCard>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { MagicStick } from '@danqing/dq-shell';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { toast } from '@/utils/feedback';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import { getItem, setItem, DQ_STORAGE } from '@/utils/storage';
import LoraTrainRunDetail from '@/components/lora/LoraTrainRunDetail.vue';
import LoraDatasetPanel from '@/components/lora/LoraDatasetPanel.vue';
import LoraTrainHistory from '@/components/lora/LoraTrainHistory.vue';
import LoraQualityHints from '@/components/lora/LoraQualityHints.vue';
import { useLoraTrainLibrary } from '@/composables/useLoraTrainLibrary';
import { useRegistryStore } from '@/stores/registry';
import type { LoraDatasetHealthReport } from '@/utils/loraQuality';
import '@/styles/editor-tabs.css';

const { t, locale } = useI18n();
const router = useRouter();
const route = useRoute();
const registryStore = useRegistryStore();

const step = ref(0);
const submitting = ref(false);
const activeRunId = ref('');
const resumeIntent = ref(false);
const applyingResumeSync = ref(false);
const trainableModels = ref<any[]>([]);
const datasets = ref<any[]>([]);
const datasetHealth = ref<LoraDatasetHealthReport | null>(null);
const datasetVlmLoading = ref(false);
const visionAvailable = ref(false);
const requirements = ref<any>(null);
const presetsByModel = ref<Record<string, Record<string, Record<string, unknown>>>>({});
const captionEdits = reactive<Record<string, string>>({});
const datasetPanelRef = ref<InstanceType<typeof LoraDatasetPanel> | null>(null);
const historyRef = ref<InstanceType<typeof LoraTrainHistory> | null>(null);
const { loading: historyLoading, refresh: refreshHistory } = useLoraTrainLibrary();
const presetKeys = computed(() => {
  if (form.base_model === 'z-image') {
    return ['scheme4', 'quick', 'standard', 'quality', 'custom'];
  }
  return ['quick', 'standard', 'quality', 'custom'];
});

const Z_IMAGE_PRIOR_DISABLED_MODELS = new Set(['z-image', 'z-image-turbo']);

const supportsPriorPreservation = computed(
  () => !Z_IMAGE_PRIOR_DISABLED_MODELS.has(form.base_model),
);

function clearPriorPreservationFields() {
  form.class_prompt = '';
  form.prior_loss_weight = null;
}

const stepNavItems = computed(() => [
  { key: 'base', label: t('loraTrain.stepBase') },
  { key: 'dataset', label: t('loraTrain.stepDataset') },
  { key: 'config', label: t('loraTrain.stepConfig') },
  { key: 'confirm', label: t('loraTrain.stepConfirm') },
]);

const form = reactive({
  base_model: 'z-image',
  dataset_id: '',
  preset: 'scheme4' as string,
  progress_prompt: '',
  default_prompt: '',
  output_name: '',
  iterations: null as number | null,
  lora_rank: null as number | null,
  learning_rate: null as number | null,
  resolution_edge: null as number | null,
  progress_steps: null as number | null,
  num_augmentations: null as number | null,
  grad_accumulate: null as number | null,
  qlora_bits: '' as '' | '4' | '8',
  grad_checkpoint: false,
  compile_step: false,
  lora_blocks: null as number | null,
  lora_alpha: null as number | null,
  lora_dropout: null as number | null,
  lora_module_keys: '',
  optimizer: '' as '' | 'adam' | 'adamw',
  weight_decay: null as number | null,
  val_split: null as number | null,
  val_every: null as number | null,
  resume_task_id: '',
  resume_checkpoint: '',
  train_type: '' as '' | 'lora' | 'dora',
  min_snr_gamma: null as number | null,
  class_prompt: '',
  prior_loss_weight: null as number | null,
  early_stop_patience: null as number | null,
  fuse_adapters: false,
});

const selectedDataset = computed(() =>
  datasets.value.find((d) => d.id === form.dataset_id)
);

const modelPresets = computed(
  () => presetsByModel.value[form.base_model] || presetsByModel.value['flux1-dev'] || {}
);

const activePresetDetail = computed(() => {
  if (form.preset === 'custom') return null;
  return (modelPresets.value[form.preset] || null) as Record<string, unknown> | null;
});

const confirmPreviewImages = computed(() =>
  (selectedDataset.value?.images || []).slice(0, 8)
);

const effectiveDatasetCaption = computed(() => {
  // form.default_prompt is the live editor value; datasets[] may lag until blur save.
  if (form.dataset_id) {
    const fromForm = form.default_prompt.trim();
    if (fromForm) return fromForm;
  }
  return String(selectedDataset.value?.default_prompt || '').trim();
});

const promptCaptionMismatch = computed(() => {
  const progress = form.progress_prompt.trim();
  const datasetCaption = effectiveDatasetCaption.value;
  if (!progress || !datasetCaption) return false;
  return progress !== datasetCaption;
});

function syncProgressPromptFromDataset() {
  const caption = effectiveDatasetCaption.value;
  if (caption) form.progress_prompt = caption;
}

const effectiveQloraBits = computed(() => {
  if (form.qlora_bits === '4') return 4;
  if (form.qlora_bits === '8') return 8;
  return null;
});

const confirmOptimizationSummary = computed(() => {
  const parts: string[] = [];
  if (effectiveQloraBits.value) {
    parts.push(t('loraTrain.summaryQlora', { bits: effectiveQloraBits.value }));
  }
  if (form.grad_checkpoint) {
    parts.push(t('loraTrain.summaryGradCkpt'));
  }
  if (form.compile_step) {
    parts.push(t('loraTrain.compileStep'));
  }
  const vs = form.val_split ?? (activePresetDetail.value?.val_split as number | undefined);
  if (vs && Number(vs) > 0) {
    parts.push(t('loraTrain.summaryValSplit', { pct: Math.round(Number(vs) * 100) }));
  }
  return parts.join(' · ');
});

const wizardSummary = computed(() => {
  const rows: Array<{ key: string; label: string; value: string }> = [];
  if (form.base_model) {
    const m = trainableModels.value.find((x) => x.id === form.base_model);
    rows.push({ key: 'base', label: t('loraTrain.summaryBase'), value: modelDisplayName(m || { id: form.base_model }) });
  }
  if (selectedDataset.value) {
    rows.push({
      key: 'dataset',
      label: t('loraTrain.summaryDataset'),
      value: `${selectedDataset.value.name} (${selectedDataset.value.image_count || 0})`,
    });
  }
  if (form.preset && step.value >= 2) {
    rows.push({
      key: 'preset',
      label: t('loraTrain.summaryPreset'),
      value: t(`loraTrain.preset.${form.preset}`),
    });
  }
  return rows;
});

function stepDone(i: number): boolean {
  if (i === 0) return !!form.base_model && step.value > 0;
  if (i === 1) return (selectedDataset.value?.image_count || 0) >= 3 && step.value > 1;
  if (i === 2) return !!form.progress_prompt.trim() && step.value > 2;
  return false;
}

function presetStats(p: string): string {
  const cfg = modelPresets.value[p] as Record<string, unknown> | undefined;
  if (!cfg) return '';
  const it = cfg.iterations;
  const rank = cfg.lora_rank;
  if (it == null || rank == null) return '';
  return t('loraTrain.presetStatsShort', { iterations: it, rank });
}

function presetResolution(cfg: Record<string, unknown> | null): string {
  const res = cfg?.resolution;
  if (!Array.isArray(res) || res.length < 2) return '';
  return `${res[0]}×${res[1]}`;
}

function presetResolutionBody(p: string): [number, number] | null {
  if (p === 'custom') return null;
  const cfg = modelPresets.value[p] as Record<string, unknown> | undefined;
  const res = cfg?.resolution;
  if (!Array.isArray(res) || res.length < 1) return null;
  const edge = Number(res[0]);
  if (!Number.isFinite(edge) || edge < 256) return null;
  const h = res.length >= 2 ? Number(res[1]) : edge;
  return [edge, Number.isFinite(h) && h >= 256 ? h : edge];
}

function confirmImageUrl(file: string): string {
  if (!form.dataset_id || !file) return '';
  return api.loras.datasetImageUrl(form.dataset_id, file);
}

const maxReachableStep = computed(() => {
  let max = 0;
  if (form.base_model) max = 1;
  if ((selectedDataset.value?.image_count || 0) >= 3) max = Math.max(max, 2);
  if (form.progress_prompt.trim()) max = Math.max(max, 3);
  return max;
});

function canNext(stepIndex: number): boolean {
  if (stepIndex === 0) return !!form.base_model;
  if (stepIndex === 1) return (selectedDataset.value?.image_count || 0) >= 3;
  if (stepIndex === 2) return !!form.progress_prompt.trim();
  return true;
}

const canSubmit = computed(
  () => canNext(2) && requirements.value?.can_submit !== false
);

const hasResume = computed(
  () => !!form.resume_task_id.trim() && !!form.resume_checkpoint.trim()
);

function clearResume() {
  form.resume_task_id = '';
  form.resume_checkpoint = '';
  resumeIntent.value = false;
  persistDraft();
}

function goToStep(i: number) {
  if (i > maxReachableStep.value) return;
  step.value = i;
}

function openRun(taskId: string) {
  if (!taskId) return;
  activeRunId.value = taskId;
}

function onRunBack() {
  activeRunId.value = '';
  historyRef.value?.refresh();
}

function modelDisplayName(m: any): string {
  const n = m.name;
  if (n && typeof n === 'object') return n.en || n.zh || m.id;
  return m.id;
}

function modelInitials(m: any): string {
  const name = modelDisplayName(m);
  const parts = name.replace(/[\[\]()]/g, ' ').split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function modelFamilyKey(id: string): string {
  if (id.startsWith('flux')) return 'flux';
  if (id.startsWith('z-image')) return 'zimage';
  if (id.startsWith('qwen')) return 'qwen';
  return 'default';
}

function modelShortDesc(m: { id?: string }): string {
  const reg = registryStore.registry?.models?.[m.id || ''];
  const desc = reg?.description;
  if (!desc || typeof desc !== 'object') return '';
  const loc = String(locale.value || 'en').startsWith('zh') ? 'zh' : 'en';
  const text = String((desc as Record<string, string>)[loc] || desc.en || desc.zh || '').trim();
  if (!text) return '';
  return text.length > 72 ? `${text.slice(0, 69)}…` : text;
}

function apiErrorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (detail && typeof detail === 'object' && detail.message) return detail.message;
  if (typeof detail === 'string') return detail;
  return err?.message || String(e);
}

function onCaptionEditsUpdate(next: Record<string, string>) {
  Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
  Object.assign(captionEdits, next);
}

async function refreshDatasetHealth(id?: string) {
  const dsId = (id || form.dataset_id || '').trim();
  if (!dsId) {
    datasetHealth.value = null;
    return;
  }
  try {
    datasetHealth.value = (await api.loras.datasetHealth(dsId)) as LoraDatasetHealthReport;
    if (datasetHealth.value?.vision_available != null) {
      visionAvailable.value = Boolean(datasetHealth.value.vision_available);
    }
  } catch {
    datasetHealth.value = null;
  }
}

async function refreshVisionAvailability() {
  try {
    const info = await api.gen.getVisionModelInfo();
    visionAvailable.value = Boolean(info?.available);
  } catch {
    visionAvailable.value = false;
  }
}

async function runDatasetVlmAudit() {
  if (!form.dataset_id || datasetVlmLoading.value) return;
  datasetVlmLoading.value = true;
  try {
    const kind = selectedDataset.value?.kind === 'style' ? 'style' : 'concept';
    datasetHealth.value = (await api.loras.datasetHealthVlm(form.dataset_id, {
      auditKind: kind,
    })) as LoraDatasetHealthReport;
    toast.success(t('loraTrain.quality.vlmAuditDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    datasetVlmLoading.value = false;
  }
}

function onDatasetsChanged(next: any[]) {
  datasets.value = next;
  void refreshDatasetHealth();
}

function selectBase(m: any) {
  if (!m.trainable || !m.ready) return;
  form.base_model = m.id;
  applyPresetTrainingDefaults();
}

function qloraBitsForApi(): number | null {
  return effectiveQloraBits.value;
}

async function loadMeta() {
  const [modelsRes, dsRes, reqRes] = await Promise.all([
    api.loras.trainableModels(),
    api.loras.listDatasets(),
    api.loras.trainingRequirements(form.base_model, qloraBitsForApi()),
  ]);
  trainableModels.value = (modelsRes as any).items || [];
  presetsByModel.value = (modelsRes as any).presets_by_model || {};
  datasets.value = (dsRes as any).items || [];
  requirements.value = reqRes;
}

async function refreshRequirements() {
  try {
    requirements.value = await api.loras.trainingRequirements(form.base_model, qloraBitsForApi());
  } catch {
    // ignore
  }
}

function applyPresetTrainingDefaults() {
  if (form.preset === 'custom') return;
  const cfg = modelPresets.value[form.preset] as Record<string, unknown> | undefined;
  form.grad_checkpoint = Boolean(cfg?.grad_checkpoint);
  if (cfg && Object.prototype.hasOwnProperty.call(cfg, 'min_snr_gamma')) {
    form.min_snr_gamma = Number(cfg.min_snr_gamma) || 0;
  } else {
    form.min_snr_gamma = null;
  }
  if (cfg && Object.prototype.hasOwnProperty.call(cfg, 'lora_blocks')) {
    form.lora_blocks = Number(cfg.lora_blocks) || null;
  } else {
    form.lora_blocks = null;
  }
  if (cfg && Object.prototype.hasOwnProperty.call(cfg, 'progress_steps')) {
    form.progress_steps = Number(cfg.progress_steps) || null;
  } else {
    form.progress_steps = null;
  }
  if (cfg && Object.prototype.hasOwnProperty.call(cfg, 'prior_loss_weight')) {
    form.prior_loss_weight = Number(cfg.prior_loss_weight) || 0;
  } else {
    form.prior_loss_weight = null;
  }
  if (!supportsPriorPreservation.value) {
    clearPriorPreservationFields();
  }
}

function applyCustomDefaultsFromQuick() {
  const quick = modelPresets.value.quick as Record<string, unknown> | undefined;
  if (!quick) return;
  if (form.iterations == null) form.iterations = Number(quick.iterations) || 400;
  if (form.lora_rank == null) form.lora_rank = Number(quick.lora_rank) || 8;
  if (form.learning_rate == null) form.learning_rate = Number(quick.learning_rate) || 0.0001;
  const res = quick.resolution as number[] | undefined;
  if (form.resolution_edge == null) form.resolution_edge = Number(res?.[0]) || 512;
  if (form.progress_steps == null) form.progress_steps = 20;
  if (form.min_snr_gamma == null) form.min_snr_gamma = 5;
  if (form.lora_blocks == null && quick.lora_blocks != null) {
    form.lora_blocks = Number(quick.lora_blocks);
  }
  if (form.num_augmentations == null) form.num_augmentations = 3;
  if (form.grad_accumulate == null) {
    form.grad_accumulate = Number(quick.grad_accumulate) || 4;
  }
  if (form.prior_loss_weight == null && supportsPriorPreservation.value) form.prior_loss_weight = 0;
  form.grad_checkpoint = true;
}

function parseModuleKeys(raw: string): string[] | undefined {
  const keys = raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return keys.length ? keys : undefined;
}

function activePresetNumber(key: string): number | null {
  if (form.preset === 'custom') return null;
  const cfg = modelPresets.value[form.preset] as Record<string, unknown> | undefined;
  if (!cfg || !Object.prototype.hasOwnProperty.call(cfg, key)) return null;
  const value = Number(cfg[key]);
  return Number.isFinite(value) ? value : null;
}

function shouldSendPresetNumberOverride(key: string, value: number | null): boolean {
  if (value == null || !Number.isFinite(Number(value))) return false;
  if (form.preset === 'custom') return true;
  const presetValue = activePresetNumber(key);
  return presetValue == null || Number(value) !== presetValue;
}

function buildTrainingBody(): Record<string, unknown> {
  const body: Record<string, unknown> = {
    base_model: form.base_model,
    dataset_id: form.dataset_id,
    progress_prompt: form.progress_prompt,
    preset: form.preset,
    output_name: form.output_name,
    auto_register: true,
  };

  if (effectiveQloraBits.value) body.qlora_bits = effectiveQloraBits.value;
  if (form.preset === 'custom') {
    body.grad_checkpoint = form.grad_checkpoint;
  } else {
    const presetGradCheckpoint = Boolean(
      (modelPresets.value[form.preset] as Record<string, unknown> | undefined)?.grad_checkpoint
    );
    if (form.grad_checkpoint !== presetGradCheckpoint) {
      body.grad_checkpoint = form.grad_checkpoint;
    }
  }
  if (form.compile_step) body.compile_step = true;

  if (form.preset === 'custom') {
    if (form.iterations) body.iterations = form.iterations;
    if (form.lora_rank) body.lora_rank = form.lora_rank;
    if (form.learning_rate) body.learning_rate = form.learning_rate;
    const edge = Number(form.resolution_edge);
    if (edge >= 256) body.resolution = [edge, edge];
    if (form.progress_steps != null && form.progress_steps >= 4) {
      body.progress_steps = form.progress_steps;
    }
    if (form.lora_blocks != null && form.lora_blocks > 0) body.lora_blocks = form.lora_blocks;
    if (form.num_augmentations != null && form.num_augmentations >= 1) {
      body.num_augmentations = form.num_augmentations;
    }
    if (form.grad_accumulate != null && form.grad_accumulate >= 1) {
      body.grad_accumulate = form.grad_accumulate;
    }
  } else {
    const presetRes = presetResolutionBody(form.preset);
    if (presetRes) body.resolution = presetRes;
    if (form.lora_blocks != null && form.lora_blocks > 0) {
      const presetBlocks = Number((modelPresets.value[form.preset] as Record<string, unknown> | undefined)?.lora_blocks);
      if (!presetBlocks || form.lora_blocks !== presetBlocks) {
        body.lora_blocks = form.lora_blocks;
      }
    }
  }
  if (form.lora_alpha != null && form.lora_alpha > 0) body.lora_alpha = form.lora_alpha;
  if (form.lora_dropout != null && form.lora_dropout >= 0) body.lora_dropout = form.lora_dropout;
  if (form.optimizer === 'adam' || form.optimizer === 'adamw') body.optimizer = form.optimizer;
  if (form.weight_decay != null && form.weight_decay >= 0) body.weight_decay = form.weight_decay;
  if (form.val_split != null && form.val_split >= 0) body.val_split = form.val_split;
  if (form.val_every != null && form.val_every > 0) body.val_every = form.val_every;

  const moduleKeys = parseModuleKeys(form.lora_module_keys);
  if (moduleKeys) body.lora_module_keys = moduleKeys;

  const resumeTask = form.resume_task_id.trim();
  const resumeCk = form.resume_checkpoint.trim();
  if (resumeTask && resumeCk) {
    body.resume_task_id = resumeTask;
    body.resume_checkpoint = resumeCk;
  }

  if (form.train_type === 'lora' || form.train_type === 'dora') body.train_type = form.train_type;
  if (form.preset === 'custom' || form.preset === 'standard' || form.preset === 'quality') {
    if (
      form.min_snr_gamma != null
      && form.min_snr_gamma >= 0
      && shouldSendPresetNumberOverride('min_snr_gamma', form.min_snr_gamma)
    ) {
      body.min_snr_gamma = form.min_snr_gamma;
    }
  }
  if (supportsPriorPreservation.value && form.class_prompt.trim()) {
    body.class_prompt = form.class_prompt.trim();
  }
  if (
    supportsPriorPreservation.value
    && form.prior_loss_weight != null
    && form.prior_loss_weight >= 0
    && shouldSendPresetNumberOverride('prior_loss_weight', form.prior_loss_weight)
  ) {
    body.prior_loss_weight = form.prior_loss_weight;
  }
  if (form.early_stop_patience != null && form.early_stop_patience > 0) {
    body.early_stop_patience = form.early_stop_patience;
  }
  if (form.fuse_adapters) body.fuse_adapters = true;

  return body;
}

function restoreDraft() {
  try {
    const raw = getItem(DQ_STORAGE.LORA_TRAIN_DRAFT);
    if (!raw) return;
    const draft = JSON.parse(raw) as Partial<typeof form> & { step?: number };
    if (draft.base_model) form.base_model = draft.base_model;
    if (draft.dataset_id) form.dataset_id = draft.dataset_id;
    if (draft.preset) form.preset = draft.preset;
    if (draft.progress_prompt) form.progress_prompt = draft.progress_prompt;
    if (draft.default_prompt) form.default_prompt = draft.default_prompt;
    if (draft.output_name) form.output_name = draft.output_name;
    if (draft.iterations != null) form.iterations = Number(draft.iterations);
    if (draft.lora_rank != null) form.lora_rank = Number(draft.lora_rank);
    if (draft.learning_rate != null) form.learning_rate = Number(draft.learning_rate);
    if (draft.resolution_edge != null) form.resolution_edge = Number(draft.resolution_edge);
    if (draft.progress_steps != null) form.progress_steps = Number(draft.progress_steps);
    if (draft.num_augmentations != null) form.num_augmentations = Number(draft.num_augmentations);
    if (draft.grad_accumulate != null) form.grad_accumulate = Number(draft.grad_accumulate);
    if (draft.qlora_bits) form.qlora_bits = draft.qlora_bits as '' | '4' | '8';
    if (typeof draft.grad_checkpoint === 'boolean') form.grad_checkpoint = draft.grad_checkpoint;
    if (typeof draft.compile_step === 'boolean') form.compile_step = draft.compile_step;
    if (draft.lora_blocks != null) form.lora_blocks = Number(draft.lora_blocks);
    if (draft.lora_alpha != null) form.lora_alpha = Number(draft.lora_alpha);
    if (draft.lora_dropout != null) form.lora_dropout = Number(draft.lora_dropout);
    if (draft.lora_module_keys) form.lora_module_keys = String(draft.lora_module_keys);
    if (draft.optimizer) form.optimizer = draft.optimizer as '' | 'adam' | 'adamw';
    if (draft.weight_decay != null) form.weight_decay = Number(draft.weight_decay);
    if (draft.val_split != null) form.val_split = Number(draft.val_split);
    if (draft.val_every != null) form.val_every = Number(draft.val_every);
    // Never restore resume fields from draft — they leak across unrelated new runs.
    form.resume_task_id = '';
    form.resume_checkpoint = '';
    resumeIntent.value = false;
    if (draft.train_type) form.train_type = draft.train_type as '' | 'lora' | 'dora';
    if (draft.min_snr_gamma != null) form.min_snr_gamma = Number(draft.min_snr_gamma);
    if (draft.class_prompt) form.class_prompt = String(draft.class_prompt);
    if (draft.prior_loss_weight != null) form.prior_loss_weight = Number(draft.prior_loss_weight);
    if (draft.early_stop_patience != null) form.early_stop_patience = Number(draft.early_stop_patience);
    if (typeof draft.fuse_adapters === 'boolean') form.fuse_adapters = draft.fuse_adapters;
    if (typeof draft.step === 'number' && draft.step >= 0 && draft.step <= 3) {
      step.value = Math.min(draft.step, maxReachableStep.value);
    }
  } catch {
    // ignore corrupt draft
  }
}

function persistDraft() {
  setItem(
    DQ_STORAGE.LORA_TRAIN_DRAFT,
    JSON.stringify({
      base_model: form.base_model,
      dataset_id: form.dataset_id,
      preset: form.preset,
      progress_prompt: form.progress_prompt,
      default_prompt: form.default_prompt,
      output_name: form.output_name,
      iterations: form.iterations,
      lora_rank: form.lora_rank,
      learning_rate: form.learning_rate,
      resolution_edge: form.resolution_edge,
      progress_steps: form.progress_steps,
      num_augmentations: form.num_augmentations,
      grad_accumulate: form.grad_accumulate,
      qlora_bits: form.qlora_bits,
      grad_checkpoint: form.grad_checkpoint,
      compile_step: form.compile_step,
      lora_blocks: form.lora_blocks,
      lora_alpha: form.lora_alpha,
      lora_dropout: form.lora_dropout,
      lora_module_keys: form.lora_module_keys,
      optimizer: form.optimizer,
      weight_decay: form.weight_decay,
      val_split: form.val_split,
      val_every: form.val_every,
      train_type: form.train_type,
      min_snr_gamma: form.min_snr_gamma,
      class_prompt: form.class_prompt,
      prior_loss_weight: form.prior_loss_weight,
      early_stop_patience: form.early_stop_patience,
      fuse_adapters: form.fuse_adapters,
      step: step.value,
    })
  );
}

watch(
  () => form.dataset_id,
  async (id) => {
    if (!id) {
      datasetHealth.value = null;
      return;
    }
    void refreshDatasetHealth(id);
    try {
      const ds = await api.loras.getDataset(id);
      Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
      for (const img of (ds as any).images || []) {
        captionEdits[img.file] = img.prompt || '';
      }
      datasets.value = datasets.value.map((d) => (d.id === id ? { ...d, ...(ds as Record<string, unknown>) } : d));
      const metaPrompt = String((ds as Record<string, unknown>).default_prompt || '').trim();
      if (metaPrompt) {
        form.default_prompt = metaPrompt;
      }
    } catch (e: unknown) {
      toast.error(apiErrorMessage(e));
    }
  }
);

async function importFromRouteQuery() {
  const raw = route.query.import;
  const ids = (Array.isArray(raw) ? raw.join(',') : String(raw || ''))
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!ids.length) return;

  const explicitDatasetId = String(route.query.dataset_id || '').trim();
  const datasetName =
    String(route.query.dataset_name || '').trim() || t('loraTrain.canvasImportDataset');

  if (explicitDatasetId) {
    form.dataset_id = explicitDatasetId;
    if (!datasets.value.some((d) => d.id === explicitDatasetId)) {
      try {
        const ds = await api.loras.getDataset(explicitDatasetId);
        datasets.value = [ds as Record<string, unknown>, ...datasets.value];
      } catch (e: unknown) {
        toast.error(apiErrorMessage(e));
        return;
      }
    }
  } else if (datasetPanelRef.value) {
    await datasetPanelRef.value.createDatasetWithName(datasetName);
  } else {
    const ds = (await api.loras.createDataset({
      name: datasetName,
      default_prompt: datasetName,
    })) as Record<string, any>;
    form.default_prompt = datasetName;
    datasets.value.unshift(ds);
    form.dataset_id = ds.id;
  }

  if (!form.dataset_id) return;
  try {
    const ds = await api.loras.importAssets(form.dataset_id, ids, form.default_prompt);
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
    for (const img of (ds as any).images || []) {
      captionEdits[img.file] = img.prompt || form.default_prompt || '';
    }
    step.value = 1;
    toast.success(t('loraTrain.galleryImported'));
    await refreshDatasetHealth(form.dataset_id);
    const q = { ...route.query };
    delete q.import;
    delete q.dataset_name;
    delete q.dataset_id;
    router.replace({ query: q });
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

function openRunFromQuery() {
  const runId = String(route.query.run || '').trim();
  if (!runId) return;
  activeRunId.value = runId;
  const q = { ...route.query };
  delete q.run;
  router.replace({ query: q });
}

function resumeFromRouteQuery() {
  const taskId = String(route.query.resume_task || '').trim();
  const checkpoint = String(route.query.resume_checkpoint || '').trim();
  if (!taskId || !checkpoint) return Promise.resolve();
  applyingResumeSync.value = true;
  form.resume_task_id = taskId;
  form.resume_checkpoint = checkpoint;
  resumeIntent.value = true;
  step.value = Math.max(step.value, 2);
  const q = { ...route.query };
  delete q.resume_task;
  delete q.resume_checkpoint;
  router.replace({ query: q });
  return api.gen.getMediaTask(taskId).then((row: any) => {
    const rec = row as Record<string, unknown>;
    const params = (rec.params || {}) as Record<string, unknown>;
    const sourceBase = String(params.base_model || rec.model_id || '').split(':')[0].trim();
    if (sourceBase) {
      form.base_model = sourceBase;
      applyPresetTrainingDefaults();
      void refreshRequirements();
    }
    toast.success(t('loraTrain.resumeTraining'));
  }).catch(() => {
    toast.success(t('loraTrain.resumeTraining'));
  }).finally(() => {
    applyingResumeSync.value = false;
  });
}

watch(step, (s) => {
  if (s === 2 && !form.progress_prompt.trim() && form.default_prompt.trim()) {
    form.progress_prompt = form.default_prompt.trim();
  }
  persistDraft();
});

watch(
  () => [
    form.base_model,
    form.dataset_id,
    form.preset,
    form.progress_prompt,
    form.default_prompt,
    form.output_name,
    form.iterations,
    form.lora_rank,
    form.learning_rate,
    form.resolution_edge,
    form.progress_steps,
    form.num_augmentations,
    form.grad_accumulate,
    form.qlora_bits,
    form.grad_checkpoint,
    form.compile_step,
    form.lora_blocks,
    form.lora_alpha,
    form.lora_dropout,
    form.lora_module_keys,
    form.optimizer,
    form.weight_decay,
    form.val_split,
    form.val_every,
  ],
  () => persistDraft()
);

watch(() => form.base_model, (mid) => {
  if (applyingResumeSync.value) return;
  if (hasResume.value) {
    const hadIntent = resumeIntent.value;
    clearResume();
    if (hadIntent) {
      toast.info(t('loraTrain.resumeClearedOnBaseChange'));
    }
  }
  if (form.preset === 'scheme4' && mid !== 'z-image') {
    form.preset = 'standard';
  }
  if (!supportsPriorPreservation.value) {
    clearPriorPreservationFields();
  }
  applyPresetTrainingDefaults();
  void refreshRequirements();
});

watch(() => form.qlora_bits, () => {
  void refreshRequirements();
});

watch(() => form.preset, (preset) => {
  applyPresetTrainingDefaults();
  if (preset === 'custom') applyCustomDefaultsFromQuick();
});

async function submitTraining() {
  submitting.value = true;
  try {
    const body = buildTrainingBody();
    const res = await api.loras.submitTraining(body);
    const tid = taskIdFromSubmitResponse(res);
    toast.success(t('loraTrain.submitted'));
    openGlobalTaskQueue();
    if (tid) {
      activeRunId.value = tid;
      historyRef.value?.refresh();
    }
    clearResume();
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    submitting.value = false;
  }
}

function startNewTraining() {
  activeRunId.value = '';
  step.value = 0;
}

function onVerifyGenerate(payload: { prompt: string; loraId: string; baseModel: string }) {
  router.push({
    name: 'image_create',
    query: {
      model: payload.baseModel,
      prompt: payload.prompt,
      lora: payload.loraId,
    },
  });
}

onMounted(async () => {
  try {
    await Promise.all([loadMeta(), refreshVisionAvailability(), registryStore.load()]);
    restoreDraft();
    if (!form.dataset_id && datasets.value.length === 1) {
      form.dataset_id = datasets.value[0].id;
    }
    applyPresetTrainingDefaults();
    if (form.preset === 'custom') applyCustomDefaultsFromQuick();
    openRunFromQuery();
    await resumeFromRouteQuery();
    await importFromRouteQuery();
    if (form.dataset_id) {
      await refreshDatasetHealth(form.dataset_id);
    }
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
});
</script>

<style scoped>
.lora-train-page__sidebar-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.lora-train-page__sidebar-card :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.lora-train-page__sidebar-card .lv-sidebar-card__new-btn {
  margin-bottom: 12px;
  flex-shrink: 0;
}

.lora-train-page__sidebar-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.lora-train-page__sidebar-body .lora-train-history {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.lora-train-page__sidebar-body .lora-train-history__list {
  max-height: none;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.lora-train-page .dq-download-menu__label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Stepper (reuses lv-editor-tabs visual) ── */
.lora-train-page__stepper {
  flex-shrink: 0;
}

.lora-train-page__stepper .lv-editor-tabs__tab--locked {
  cursor: not-allowed;
  opacity: 0.45;
}

/* ── Nav bar (panel bottom) ── */
.lora-train-page__nav-bar {
 display: flex;
 align-items: center;
 justify-content: space-between;
 padding: 10px 0 0;
 border-top: 0.5px solid var(--dq-border-subtle);
 flex-shrink: 0;
 margin-top: auto;
 width: 100%;
 box-sizing: border-box;
}

.lora-train-page .copilot-page__workspace-card > :deep(.dq-surface-card__body) {
 flex: 1;
 min-height: 0;
 overflow: hidden;
 display: flex;
 flex-direction: column;
 padding: 12px 16px 16px;
 width: 100%;
 max-width: 100%;
 box-sizing: border-box;
}

.lora-train-page__panel {
  width: 100%;
  min-width: 0;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.lora-train-page__panel:not(.lora-train-page__panel--base) {
  overflow-y: auto;
  overflow-x: hidden;
}

.lora-train-page__panel--base {
  width: 100%;
  overflow-y: auto;
  overflow-x: hidden;
}

.lora-train-page__panel--flush {
  max-width: none;
  width: 100%;
}

.lora-train-page__base-grid {
 display: grid;
 grid-template-columns: repeat(2, minmax(0, 1fr));
 gap: 16px;
 width: 100%;
 align-items: stretch;
}

.lora-train-page__base-tile {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  width: 100%;
  min-width: 0;
  height: 100%;
  box-sizing: border-box;
  padding: 20px;
  border: 1.5px solid var(--dq-border-subtle);
  border-radius: 16px;
  background: var(--dq-bg-base);
  cursor: pointer;
  text-align: left;
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease,
    transform 0.15s ease;
}

.lora-train-page__base-tile:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.lora-train-page__base-tile:not(:disabled):hover {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 3%, var(--dq-bg-base));
  transform: translateY(-2px);
  box-shadow: 0 8px 24px color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lora-train-page__base-tile.is-selected {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 6%, var(--dq-bg-base));
  box-shadow:
    inset 0 0 0 1.5px var(--dq-accent),
    0 12px 32px color-mix(in srgb, var(--dq-accent) 12%, transparent);
}

.lora-train-page__base-tile-icon {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: 14px;
  font-size: var(--dq-font-size-title);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-tighter);
  color: var(--dq-label-primary);
  background: var(--dq-fill-secondary);
  border: 1px solid var(--dq-border-subtle);
  transition: all 0.2s ease;
}

.lora-train-page__base-tile.is-family-flux .lora-train-page__base-tile-icon {
  color: #c4b5fd;
  background: linear-gradient(135deg, color-mix(in srgb, #8b5cf6 15%, var(--dq-fill-secondary)), color-mix(in srgb, #8b5cf6 8%, var(--dq-fill-secondary)));
  border-color: color-mix(in srgb, #8b5cf6 30%, transparent);
}

.lora-train-page__base-tile.is-family-zimage .lora-train-page__base-tile-icon {
  color: #67e8f9;
  background: linear-gradient(135deg, color-mix(in srgb, #06b6d4 15%, var(--dq-fill-secondary)), color-mix(in srgb, #06b6d4 8%, var(--dq-fill-secondary)));
  border-color: color-mix(in srgb, #06b6d4 30%, transparent);
}

.lora-train-page__base-tile.is-family-qwen .lora-train-page__base-tile-icon {
  color: #fdba74;
  background: linear-gradient(135deg, color-mix(in srgb, #f97316 15%, var(--dq-fill-secondary)), color-mix(in srgb, #f97316 8%, var(--dq-fill-secondary)));
  border-color: color-mix(in srgb, #f97316 30%, transparent);
}

.lora-train-page__base-tile.is-selected .lora-train-page__base-tile-icon {
  transform: scale(1.05);
  box-shadow: 0 4px 12px color-mix(in srgb, var(--dq-accent) 20%, transparent);
}

.lora-train-page__base-tile-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  padding-top: 4px;
}

.lora-train-page__base-tile-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-tight);
  color: var(--dq-label-primary);
  line-height: 1.3;
}

.lora-train-page__base-tile-desc {
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.lora-train-page__base-tile-check {
  position: absolute;
  right: 16px;
  bottom: 16px;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: var(--dq-accent);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--dq-accent) 15%, transparent);
}

.lora-train-page__base-tile-check::after {
  content: '';
  position: absolute;
  left: 7px;
  top: 3px;
  width: 6px;
  height: 10px;
  border: solid var(--dq-on-accent);
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.lora-train-page__panel--flush > :deep(.lora-dataset-panel) {
}

.lora-train-page__preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
  width: 100%;
}

.lora-train-page__preset-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 14px 16px;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  cursor: pointer;
  text-align: left;
  transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}

.lora-train-page__preset-card:hover {
  border-color: var(--dq-border);
}

.lora-train-page__preset-card.is-recommended {
  border-color: var(--dq-accent-border-hover);
}

.lora-train-page__preset-badge {
  margin-left: 0.35rem;
  font-size: var(--dq-font-size-caption);
  font-style: normal;
  font-weight: 600;
  color: var(--dq-accent);
}

.lora-train-page__scheme4-hint {
  margin-bottom: 1rem;
}

.lora-train-page__preset-card.is-selected {
  border-color: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 9%, var(--dq-fill-secondary));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.lora-train-page__preset-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-train-page__preset-desc {
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-train-page__preset-stats {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-accent);
  margin-top: 2px;
}

.lora-train-page__preset-detail {
  margin-bottom: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-train-page__custom-block {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lora-train-page__custom-guide-text {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-train-page__custom-guide-list {
  margin: 0;
  padding-left: 1.1rem;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-train-page__custom-guide-list li {
  margin-bottom: 4px;
}

.lora-train-page__prompt-warn {
  margin-top: 10px;
}

.lora-train-page__prompt-warn p {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

.lora-train-page__preset-detail-label {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__preset-detail-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
}

.lora-train-page__preset-crop-hint {
  font-style: normal;
  color: var(--dq-label-tertiary);
}

.lora-train-page__field-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-train-page__confirm-intro {
  margin: 0 0 14px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-secondary);
  line-height: 1.5;
}

.lora-train-page__confirm-health {
  margin-bottom: 14px;
  width: 100%;
}

.lora-train-page__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
  min-width: 0;
}

.lora-train-page__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  color: var(--dq-label-secondary);
}

.lora-train-page__optimization {
  margin: 16px 0;
  padding: 14px 16px;
  border-radius: 12px;
  border: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  max-width: 100%;
}

.lora-train-page__optimization-head {
  margin-bottom: 12px;
}

.lora-train-page__field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px 16px;
  max-width: 100%;
}

.lora-train-page__field--wide {
  grid-column: 1 / -1;
}

.lora-train-page__field--switch {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-train-page__resume-banner {
  margin-bottom: 16px;
}

.lora-train-page__resume-banner p {
  margin: 0 0 10px;
}

.lora-train-page__advanced {
  margin-top: 4px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-secondary);
}

.lora-train-page__advanced summary {
  cursor: pointer;
  margin-bottom: 12px;
}

.lora-train-page__summary {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.lora-train-page__summary li {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-body);
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

.lora-train-page__summary span {
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.lora-train-page__summary strong {
  color: var(--dq-label-primary);
  font-weight: 600;
  text-align: right;
  word-break: break-word;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 1;
}

.lora-train-page__confirm-preview {
  margin-bottom: 16px;
  width: 100%;
}

.lora-train-page__confirm-preview-label {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__confirm-preview-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.lora-train-page__confirm-preview-grid img {
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: 10px;
  border: 0.5px solid var(--dq-border);
}

.lora-train-page__confirm-preview-more {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 64px;
  border-radius: 10px;
  background: var(--dq-fill-tertiary);
  border: 0.5px dashed var(--dq-border);
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

@media (max-width: 960px) {
  .lora-train-page.copilot-page {
    flex-direction: column;
    overflow: auto;
  }

  .lora-train-page__sidebar-history {
    flex: 0 0 auto;
    max-height: 220px;
  }

  .lora-train-page__memory-panel {
    margin-top: 0;
  }

  .lora-train-page__base-grid {
    grid-template-columns: 1fr;
  }

  .lora-train-page__stepper .lv-editor-tabs__desc {
    display: none;
  }
}
</style>
