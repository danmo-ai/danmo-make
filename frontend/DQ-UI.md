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
| Layout CSS | Legacy layout in `styles/theme.css`; shared product UI in `styles/productivity.css` |

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
- **共享工作区**：模型、提示词、设置、助手和训练页使用 `components/workspace/WorkspaceLayout.vue`、`WorkspaceSidebar.vue`、`WorkspaceHeader.vue`；二级导航保持平面，不再新增卡片式页面外壳。
- **应用导航**：展开 208px / 紧凑 64px；折叠状态存于 `DQ_STORAGE.NAV_COLLAPSED`。当前路由使用 `aria-current="page"`，紧凑模式保留 tooltip。
- **创作页**：页头只保留一个主创作按钮，不再添加重复悬浮 CTA；选择模式使用独立批量工具条。Composer 共用底部操作区，作品卡只直接展示高频动作，其余进入更多菜单。
- **交互状态**：Hover 与 `:focus-within` 必须等价；可点击元素使用原生 button，或补齐角色、tabindex、Enter/Space 和 ARIA 状态。禁止只在 Hover 时暴露关键操作。
- **动效**：常规反馈 120–180ms，不使用 `transition: all`、装饰性持续脉冲或普通工作区氛围渐变；必须兼容 `prefers-reduced-motion` 和 reduced-transparency。
- **表面**：普通工作区以中性背景、细分隔线和轻边框为主；玻璃只用于有实际层级的浮动弹层，强调色只用于主操作、当前项和状态。

### Theme CSS

Import palettes in `frontend/src/main.ts`; switch with `applyDqTheme` on `<html>`.

Studio-only chrome（侧栏、创作页工具条、gallery）留在产品层 `frontend/src/styles/productivity.css`；历史布局仍在 `theme.css`，新增页面不要继续向旧文件叠加覆盖。共享布局规则不进 tokens。

`make check-consistency` runs `check_frontend_governance.py` (EP boundary, theme legacy, dq-ui compat).

## Local dev

```bash
cd ../dq-ui && pnpm install
cd frontend && npm install && npm run dev
```

Restart Vite after `dq-ui` changes；tokens/ui 变更后建议在 `dq-ui/packages/*` 执行 `npm run build`。
