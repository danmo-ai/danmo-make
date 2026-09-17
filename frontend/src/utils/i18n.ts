import { createI18n } from 'vue-i18n';
import type { I18nOptions } from 'vue-i18n';
import zhMessages from '@/locales/zh.json';
import enMessages from '@/locales/en.json';

const messages = {
  zh: zhMessages,
  en: enMessages,
} as const;

const options: I18nOptions = {
  locale: 'zh',
  fallbackLocale: 'en',
  messages: messages as unknown as I18nOptions['messages'],
  legacy: false,
  missing: (_locale: string, key: string) => {
    console.warn(`Missing translation: ${key}`);
    return key;
  },
};

const i18n = createI18n(options);

export default i18n;

/** Get current locale string */
export function getLocale(): string {
  return (i18n.global.locale as unknown as { value: string }).value;
}

/** Typed translation helper with params */
export function $tt(key: string, params?: Record<string, string | number>): string {
  try {
    const result = i18n.global.t(key, params || {});
    return result || key;
  } catch {
    return key;
  }
}

/** Bilingual model name */
export function $mn(model: { name?: string | { zh?: string; en?: string }; name_en?: string } | null, defaultName?: string): string {
  if (!model) return defaultName || '';
  const locale = getLocale();
  const n = model.name;
  if (n && typeof n === 'object') {
    return locale === 'en' ? (n.en || n.zh || defaultName || '') : (n.zh || n.en || defaultName || '');
  }
  if (locale === 'en' && model.name_en) return model.name_en;
  return (n as string) || defaultName || '';
}

/** Bilingual model description */
export function $md(model: { description?: string | { zh?: string; en?: string }; description_en?: string } | null, defaultDesc?: string): string {
  if (!model) return defaultDesc || '';
  const locale = getLocale();
  const d = model.description;
  if (d && typeof d === 'object') {
    return locale === 'en' ? (d.en || d.zh || defaultDesc || '') : (d.zh || d.en || defaultDesc || '');
  }
  if (locale === 'en' && model.description_en) return model.description_en;
  return (d as string) || defaultDesc || '';
}

type BilingualText = string | { zh?: string; en?: string };

/** Version row label from registry ``versions.*.name`` (bilingual object or string). */
export function $vn(
  versionConfig?: { name?: BilingualText } | null,
  fallback = ''
): string {
  const vn = versionConfig?.name;
  if (vn == null || vn === '') return fallback;
  if (typeof vn === 'object') {
    const locale = getLocale();
    return locale === 'en' ? (vn.en || vn.zh || fallback) : (vn.zh || vn.en || fallback);
  }
  return String(vn);
}

/** Model + version name */
export function $mvn(
  modelKey: string,
  config: { name?: string | { zh?: string; en?: string }; name_en?: string } | null,
  versionConfig?: { name?: string | { zh?: string; en?: string } }
): string {
  const base = $mn(config, modelKey);
  const suffix = $vn(versionConfig);
  if (!suffix) return base;
  if (suffix.trim().toLowerCase() === base.trim().toLowerCase()) return base;
  return `${base} - ${suffix}`;
}

/** Preset name */
export function $pn(
  presetData: { name?: string | { zh?: string; en?: string }; name_en?: string } | null,
  chineseName?: string,
): string {
  const locale = getLocale();
  const n = presetData?.name;
  if (n && typeof n === 'object') {
    return locale === 'en' ? (n.en || n.zh || chineseName || '') : (n.zh || n.en || chineseName || '');
  }
  if (locale === 'en' && presetData?.name_en) return presetData.name_en;
  return chineseName || (typeof n === 'string' ? n : '') || presetData?.name_en || '';
}

/** 创作页主按钮下方快捷键提示（macOS 与 Windows 分文案）。 */
export function sendShortcutHintText(): string {
  const ua =
    typeof navigator !== 'undefined'
      ? navigator.platform || navigator.userAgent || ''
      : '';
  const isApple = /Mac|iPhone|iPad|iPod/i.test(ua);
  return isApple ? $tt('studio.sendShortcutHintMac') : $tt('studio.sendShortcutHintWin');
}
