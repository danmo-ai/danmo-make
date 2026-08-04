# Danmo Make UI

Shared packages live in the sibling repo [danmo-ai/dq-ui](https://github.com/danmo-ai/dq-ui) (`file:../../dq-ui/packages/*` in `package.json`).

GitHub Actions checks out `dq-ui` next to the repo so `npm install` resolves those paths (see `.github/workflows/release.yml`).

## Feedback

```ts
import { toast, confirm } from '@/utils/feedback';

toast.success('Saved');
toast.error('Failed');
await confirm('Delete this item?', 'Confirm', { type: 'warning' });
```

- Global errors: `toast.notify({ title, message })` in `main.ts`
- Loading overlay: `v-dq-loading="isLoading"`

Hosts mount via `installDanQingFeedback` in `plugins/dq-ui.ts`.

## UI stack

| Layer | Source |
|-------|--------|
| Tokens | `@danqing/dq-tokens` (`--dq-*`) |
| Components | `@danqing/dq-ui` (`Dq*`, Reka UI) |
| Shell | `@danqing/dq-shell` |
| Icons | Lucide via `registerDqIcons` + `DqIcon` |
| Layout CSS | `studio-*` / `settings-*` / `copilot-*` in `styles/theme.css` |

## Conventions

- **主题切换**：`stores/theme.ts` → `applyDqTheme` / `THEME_OPTIONS`（`@danqing/dq-tokens`）。设置页只展示 tokens 现有 **5** 套：`mac` · `mac-light` · `tokyo-night` · `nord-dark` · `minimal-light`。默认 **`mac`**。已移除主题经 `resolveDqThemeSlug` / `REMOVED_THEME_FALLBACKS` 迁移；旧值 `apple-dark` → `mac`。
- **间距 / 半径 / 配色**：只用 `--dq-*` tokens（禁止产品层 `--primary` / `--bg-*` / `--text-*` 兼容别名）。
- **字号（4 档，主题无关）**：`caption` 12 / `body`·`prose` 14 / `title` 16。禁止硬编码 9–11px 或自造字号阶梯；分区标题用 `body`（或 legacy `--dq-font-size-heading`），页面标题才用 `title`。字重/颜色做层级，勿堆叠字号。
- **字体**：`var(--dq-font-sans)` / `var(--dq-font-mono)`（或 `inherit`）；禁止产品层再写 `ui-monospace, SF Mono…` 栈。
- **Size**：紧凑控件只用 `size="sm"`（禁止 `small` / `mini` / `xs`）。
- **Select**：工具栏 / gallery 滤镜 / 画布会话用 `size="sm" variant="ghost"`；设置表单保持 default。空选项可用 `value=""`（DqOption 映射为内部 sentinel）。有选项时 v-model 应落到合法值（含默认首项），勿长期空白 placeholder。
- **Agent tokens**：`main.ts` 引入 `@danqing/dq-tokens/dq-agent.css`（`.dq-prose` / `.dq-status-dot` / `.dq-kbd`）。状态点用 `.dq-status-dot`，勿自造平行实现。
- **焦点 / 悬停**：`--dq-focus-ring`、`.dq-hoverable`；禁止自造 focus ring。
- **禁止**全局 `html * { transition: ... }`。
- 模板仅使用 `Dq*`（无 Element Plus）。

### Theme CSS

Import palettes in `frontend/src/main.ts`; switch with `applyDqTheme` on `<html>`.

Studio-only chrome（侧栏、创作页浮动条、gallery）留在 `frontend/src/styles/theme.css` — 不进 tokens。

`make check-consistency` runs `check_frontend_governance.py` (EP boundary, theme legacy, dq-ui compat).

## Local dev

```bash
cd ../dq-ui && pnpm install
cd frontend && npm install && npm run dev
```

Restart Vite after `dq-ui` changes；tokens/ui 变更后建议在 `dq-ui/packages/*` 执行 `npm run build`。
