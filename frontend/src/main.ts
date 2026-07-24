import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { registerDqIcons } from '@danqing/dq-shell';
import router from './router';
import i18n, { $tt, $mn, $md, $mvn, $pn, sendShortcutHintText } from './utils/i18n';
import { toast } from './utils/feedback';
import App from './App.vue';
import { installDanQingUi } from './plugins/dq-ui';
import { installTauriMacosShell } from './utils/desktop';
import { useThemeStore } from './stores/theme';
import '@danqing/dq-tokens/dq-mac.css';
import '@danqing/dq-tokens/dq-linear-dark.css';
import '@danqing/dq-tokens/dq-china-red-dark.css';
import '@danqing/dq-tokens/dq-shadcn-dark.css';
import '@danqing/dq-tokens/dq-shadcn-light.css';
import '@danqing/dq-tokens/dq-catppuccin.css';
import '@danqing/dq-tokens/dq-tokyo-night.css';
import '@danqing/dq-tokens/dq-minimal-light.css';
import '@danqing/dq-tokens/dq-dracula.css';
import '@danqing/dq-tokens/dq-nord-dark.css';
import '@danqing/dq-tokens/dq-catppuccin-latte.css';
import '@danqing/dq-tokens/dq-nord-light.css';
import '@danqing/dq-tokens/dq-github-light.css';
import '@danqing/dq-tokens/dq-glass.css';
import '@danqing/dq-tokens/dq-agent.css';
import '@danqing/dq-ui/style.css';
import '@danqing/dq-shell/style.css';
import './styles/theme.css';
import '@danqing/dq-tokens/dq-tauri-macos.css';

// Default flash before Pinia init — macOS dark
document.documentElement.classList.add('dq-mac', 'dark');
installTauriMacosShell();

const app = createApp(App);

registerDqIcons(app);

// Global properties
app.config.globalProperties.$tt = $tt;
app.config.globalProperties.$mn = $mn;
app.config.globalProperties.$md = $md;
app.config.globalProperties.$mvn = $mvn;
app.config.globalProperties.$pn = $pn;
app.config.globalProperties.$sendShortcutHint = sendShortcutHintText;

app.use(createPinia());
app.use(router);
installDanQingUi(app);
app.use(i18n);

const themeStore = useThemeStore();
themeStore.init();

app.mount('#app');

function notifyGlobalError(title: string, message: string) {
  toast.notify({ title, message, type: 'error', duration: 6000 });
}

// Global error handlers
app.config.errorHandler = (err) => {
  console.error('Vue error:', err);
  notifyGlobalError(
    $tt('studio.globalErrorTitle'),
    String((err as Error)?.message || err).substring(0, 200),
  );
};

window.onerror = function (msg, _source, _line, _col, error) {
  const message = (error && error.message) || String(msg);
  if (message.indexOf('ResizeObserver loop') !== -1) return true;
  notifyGlobalError($tt('studio.jsErrorTitle'), String(message).substring(0, 200));
  return false;
};

window.addEventListener('unhandledrejection', function (event) {
  const reason =
    (event.reason && event.reason.message) || event.reason || 'Unknown Promise error';
  notifyGlobalError($tt('studio.unhandledPromiseTitle'), String(reason).substring(0, 200));
});
