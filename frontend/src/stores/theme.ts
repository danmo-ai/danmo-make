import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import {
  THEME_OPTIONS as DQ_THEME_OPTIONS,
  applyDqTheme,
  getDqThemeMeta,
  resolveDqThemeSlug,
  type DqThemeSlug,
} from '@danqing/dq-tokens'
import { getItem, setItem, DQ_STORAGE } from '@/utils/storage'
import { syncTauriWindowTheme } from '@/utils/desktop'

export type ThemeId = DqThemeSlug

export interface ThemeOption {
  id: ThemeId
  label: string
  description: string
  htmlClass: string
  accent: string
  dark: boolean
}

/** Product Settings catalog — sourced from @danqing/dq-tokens. */
export const THEME_OPTIONS: ThemeOption[] = DQ_THEME_OPTIONS.map((opt) => ({
  id: opt.slug,
  label: opt.label,
  description: opt.description,
  htmlClass: opt.htmlClass,
  accent: opt.accent,
  dark: opt.dark,
}))

export const VALID_THEME_IDS: ThemeId[] = THEME_OPTIONS.map((o) => o.id)

/** Non-default themes (used by remount re-apply paths). */
export const PRODUCTIVITY_THEME_IDS: ThemeId[] = VALID_THEME_IDS.filter((id) => id !== 'mac')

/** Map legacy Studio ids → current dq-tokens slugs (5 curated themes). */
export function migrateThemeId(raw: string | null | undefined): ThemeId | null {
  if (!raw) return null
  // Pre-tokens Studio id (not in REMOVED_THEME_FALLBACKS).
  if (raw === 'apple-dark') return 'mac'
  return resolveDqThemeSlug(raw)
}

export function applyTheme(themeId?: ThemeId | string | null): void {
  const id = migrateThemeId(themeId ?? undefined) ?? 'mac'
  applyDqTheme(id)
  const dark = getDqThemeMeta(id)?.dark ?? true
  syncTauriWindowTheme(dark)
}

function getStoredTheme(): ThemeId {
  return migrateThemeId(getItem(DQ_STORAGE.THEME)) ?? 'mac'
}

export const useThemeStore = defineStore('theme', () => {
  const currentTheme = ref<ThemeId>(getStoredTheme())

  function setTheme(id: ThemeId) {
    const next = migrateThemeId(id) ?? 'mac'
    currentTheme.value = next
    applyTheme(next)
    setItem(DQ_STORAGE.THEME, next)
  }

  function init() {
    applyTheme(currentTheme.value)
    setItem(DQ_STORAGE.THEME, currentTheme.value)
  }

  watch(currentTheme, (id) => {
    applyTheme(id)
    setItem(DQ_STORAGE.THEME, id)
  })

  return { currentTheme, setTheme, init }
})
