# DanQing UI in Studio

Shared packages live in the sibling repo [danqing-ai/dq-ui](https://github.com/danqing-ai/dq-ui) (`file:../../dq-ui/packages/*` in `package.json`).

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
| Layout CSS | `studio-*` / `settings-*` / `copilot-*` in `styles/theme.css` (+ `long-video.css`) |

## Conventions

- **主题切换**：`stores/theme.ts` → `applyDqTheme` / `THEME_OPTIONS`（`@danqing/dq-tokens`）。默认 **`mac`**（macOS 暗色）。旧值 `apple-dark` 自动迁移为 `mac`。
- **间距 / 半径**：优先 `--dq-space-*`、`--dq-radius-*`；产品兼容别名 `--primary` / `--bg-*` / `--radius-*` 仅作过渡。
- **Size**：紧凑控件只用 `size="sm"`（禁止 `small` / `mini`）。
- **Select**：工具栏紧凑选择器用 `size="sm" variant="ghost"`。
- **Agent tokens**：`main.ts` 引入 `@danqing/dq-tokens/dq-agent.css`（`.dq-prose` / `.dq-status-dot` / `.dq-kbd`）。状态点用 `.dq-status-dot`，勿自造平行实现。
- **焦点 / 悬停**：`--dq-focus-ring`、`.dq-hoverable`；禁止自造 focus ring。
- **禁止**全局 `html * { transition: ... }`。
- 模板仅使用 `Dq*`（无 Element Plus）。

### Theme CSS

Import palettes in `frontend/src/main.ts`; switch with `applyDqTheme` on `<html>`.

Studio-only chrome（侧栏、创作页浮动条、gallery、长视频工作台）留在 `frontend/src/styles/theme.css` / `long-video.css` — 不进 tokens。

`make check-consistency` runs `check_frontend_governance.py` (EP boundary, theme legacy, dq-ui compat).

## Local dev

```bash
cd ../dq-ui && pnpm install
cd frontend && npm install && npm run dev
```

Restart Vite after `dq-ui` changes；tokens/ui 变更后建议在 `dq-ui/packages/*` 执行 `npm run build`。
