# Dark Mode Toggle for Graz Restaurant Chatbot

## TL;DR

> **Quick Summary**: Add a dark mode toggle to the navbar that persists user preference in localStorage and respects OS `prefers-color-scheme`. Refactor hardcoded CSS colors to custom properties first, then layer dark theme overrides on top.
>
> **Deliverables**:
> - CSS custom properties refactor (no visual change in light mode)
> - Dark theme color overrides via `[data-theme="dark"]`
> - Sun/moon toggle button in navbar
> - Dedicated `theme.js` for toggle logic + localStorage persistence
> - Inline flash-prevention script in `<head>`
>
> **Estimated Effort**: Short (~1-2 hours)
> **Parallel Execution**: NO - sequential (Task 1 must complete before Task 2)
> **Critical Path**: Task 1 (CSS refactor) -> Task 2 (dark mode feature)

---

## Context

### Original Request
User asked to add a dark mode toggle feature to the website.

### Interview Summary
**Key Discussions**:
- Toggle placement: Navbar (sun/moon icon button)
- Persistence: localStorage with OS preference as fallback
- Mechanism: `data-theme="dark"` attribute on `<html>` element

**Metis Review Findings** (addressed):
- `chat.js` is the wrong place for theme logic — use a dedicated `theme.js` file
- Navbar needs flexbox layout to accommodate the toggle button (currently no layout styles)
- `.restaurant-card h4` color `#2c3e50` will be invisible on dark background — must override
- No link color defined — browser default blue links have poor contrast in dark mode
- FOUC-prevention inline script conflicts with AGENTS.md "no inline scripts" rule — acknowledged as necessary exception (one-liner in `<head>`, not logic)
- Should split into two commits: CSS variable refactor first, then dark mode feature

---

## Work Objectives

### Core Objective
Add a fully functional dark mode toggle to the Graz Restaurant Chatbot website with smooth transitions, persistent preference, and no visual regressions.

### Concrete Deliverables
- Modified `app/static/css/style.css` — refactored to CSS custom properties with dark theme overrides
- Modified `app/templates/base.html` — navbar toggle button + inline flash-prevention script + theme.js include
- New `app/static/js/theme.js` — toggle logic, localStorage, OS preference detection
- No backend changes

### Definition of Done
- [ ] Light mode looks identical to current design (no visual regression)
- [ ] Dark mode applies consistently to all UI elements
- [ ] Toggle button visible in navbar, works on click
- [ ] Theme persists across page reloads via localStorage
- [ ] OS dark mode preference respected when no saved preference exists
- [ ] No flash of wrong theme on page load
- [ ] Smooth color transitions when toggling

### Must Have
- CSS custom properties for ALL color values (body, nav, chat, cards, input, footer)
- `[data-theme="dark"]` selector overriding all custom properties
- Toggle button with sun/moon icons visible in both themes
- localStorage read/write for theme preference
- `prefers-color-scheme: dark` media query fallback
- Link colors defined for both themes
- Restaurant card heading color overridden for dark mode
- Smooth `transition` on `background-color` and `color` properties

### Must NOT Have (Guardrails)
- NO changes to backend Python files
- NO JavaScript frameworks or libraries
- NO CSS preprocessors (keep vanilla CSS)
- NO changes to chat functionality or API logic
- NO theme logic mixed into `chat.js` — separate `theme.js` only
- NO complex animations beyond simple color transitions
- NO breaking of existing light mode appearance

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: NO (frontend-only CSS/JS change, no backend impact)
- **Framework**: N/A

### Agent-Executed QA Scenarios (MANDATORY)

**Verification Tool**: Playwright (playwright skill)

---

## Execution Strategy

### Sequential Execution (2 tasks)

```
Task 1: Refactor CSS to custom properties (no visual change)
    |
    v
Task 2: Add dark mode toggle (HTML + JS + dark CSS overrides)
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | None |
| 2 | 1 | None | None |

---

## TODOs

- [ ] 1. Refactor CSS Colors to Custom Properties

  **What to do**:
  - Open `app/static/css/style.css`
  - Add a `:root` block at the top defining CSS custom properties for every hardcoded color:
    ```css
    :root {
      --color-body-text: #333;
      --color-body-bg: #f5f5f5;
      --color-navbar-bg: #2c3e50;
      --color-navbar-text: #ffffff;
      --color-chat-container-bg: #ffffff;
      --color-chat-shadow: rgba(0,0,0,0.1);
      --color-chat-header-start: #667eea;
      --color-chat-header-end: #764ba2;
      --color-chat-messages-bg: #fafafa;
      --color-user-bubble-bg: #667eea;
      --color-user-bubble-text: #ffffff;
      --color-assistant-bubble-bg: #ffffff;
      --color-assistant-bubble-border: #e0e0e0;
      --color-assistant-bubble-text: #333;
      --color-restaurant-card-bg: #f9f9f9;
      --color-restaurant-card-border: #667eea;
      --color-restaurant-card-heading: #2c3e50;
      --color-cuisine-text: #667eea;
      --color-rating-text: #f39c12;
      --color-input-bg: #ffffff;
      --color-input-border: #e0e0e0;
      --color-input-focus-border: #667eea;
      --color-input-text: #333;
      --color-button-bg: #667eea;
      --color-button-hover-bg: #5568d3;
      --color-button-disabled-bg: #cccccc;
      --color-loading-text: #667eea;
      --color-footer-bg: #2c3e50;
      --color-footer-text: #ffffff;
      --color-link: #667eea;
    }
    ```
  - Replace every hardcoded color in the file with its corresponding `var(--color-*)` reference
  - Add `color: var(--color-link);` rule for `a` tags
  - Add `transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;` to `body`, `.chat-container`, `.chat-messages`, `.assistant-message .message-content`, `.restaurant-card`, `.chat-input-container`, `#user-input`, `.footer`
  - **CRITICAL**: After this task, the site must look EXACTLY the same as before (same colors in light mode)

  **Must NOT do**:
  - Do NOT add any dark mode styles yet
  - Do NOT change any layout or spacing
  - Do NOT touch HTML or JS files
  - Do NOT add new classes or elements

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file mechanical refactor with clear before/after
  - **Skills**: [`frontend-ui-ux`]
    - `frontend-ui-ux`: CSS custom properties expertise, visual regression awareness

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `app/static/css/style.css:1-241` — The entire current CSS file; every color value must be extracted

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Light mode visual regression check
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running on localhost:8000
    Steps:
      1. Navigate to: http://localhost:8000
      2. Wait for: .chat-container visible (timeout: 5s)
      3. Assert: body background-color computes to rgb(245, 245, 245)
      4. Assert: .navbar background-color computes to rgb(44, 62, 80)
      5. Assert: .chat-container background-color computes to rgb(255, 255, 255)
      6. Assert: .chat-messages background-color computes to rgb(250, 250, 250)
      7. Assert: #send-button background-color computes to rgb(102, 126, 234)
      8. Assert: .footer background-color computes to rgb(44, 62, 80)
      9. Screenshot: .sisyphus/evidence/task-1-light-mode-after-refactor.png
    Expected Result: All colors identical to before refactor
    Evidence: .sisyphus/evidence/task-1-light-mode-after-refactor.png
  ```

  **Evidence to Capture:**
  - [ ] Screenshot: .sisyphus/evidence/task-1-light-mode-after-refactor.png

  **Commit**: YES
  - Message: `refactor(css): extract hardcoded colors to CSS custom properties`
  - Files: `app/static/css/style.css`
  - Pre-commit: Visual comparison via Playwright screenshot

---

- [ ] 2. Add Dark Mode Toggle (HTML + JS + Dark CSS Overrides)

  **What to do**:

  **A) CSS dark theme overrides** (`app/static/css/style.css`):
  - Add `[data-theme="dark"]` block after `:root` that overrides all custom properties with dark equivalents:
    ```css
    [data-theme="dark"] {
      --color-body-text: #e0e0e0;
      --color-body-bg: #1a1a2e;
      --color-navbar-bg: #16213e;
      --color-navbar-text: #e0e0e0;
      --color-chat-container-bg: #1e1e3a;
      --color-chat-shadow: rgba(0,0,0,0.3);
      --color-chat-messages-bg: #16213e;
      --color-user-bubble-bg: #667eea;
      --color-user-bubble-text: #ffffff;
      --color-assistant-bubble-bg: #2a2a4a;
      --color-assistant-bubble-border: #3a3a5c;
      --color-assistant-bubble-text: #e0e0e0;
      --color-restaurant-card-bg: #2a2a4a;
      --color-restaurant-card-border: #667eea;
      --color-restaurant-card-heading: #e8e8e8;
      --color-cuisine-text: #8fa4f0;
      --color-rating-text: #f5b041;
      --color-input-bg: #2a2a4a;
      --color-input-border: #3a3a5c;
      --color-input-focus-border: #667eea;
      --color-input-text: #e0e0e0;
      --color-button-bg: #667eea;
      --color-button-hover-bg: #5568d3;
      --color-button-disabled-bg: #4a4a6a;
      --color-loading-text: #8fa4f0;
      --color-footer-bg: #16213e;
      --color-footer-text: #b0b0b0;
      --color-link: #8fa4f0;
    }
    ```
  - Add styles for `.navbar-inner` (flexbox row, space-between, align-center)
  - Add styles for `.theme-toggle` button:
    - Background: transparent, no border
    - Font-size: ~1.5rem for the icon
    - Color: white (navbar text)
    - Cursor: pointer
    - Show `.theme-icon--light` by default, hide `.theme-icon--dark`
    - In `[data-theme="dark"]`: hide `.theme-icon--light`, show `.theme-icon--dark`

  **B) HTML changes** (`app/templates/base.html`):
  - Restructure navbar `.container` to use `.navbar-inner` class with flexbox
  - Add toggle button inside navbar:
    ```html
    <button id="theme-toggle" class="theme-toggle" type="button"
            aria-label="Toggle dark mode">
        <span class="theme-icon theme-icon--light">&#9790;</span>
        <span class="theme-icon theme-icon--dark">&#9788;</span>
    </button>
    ```
  - Add inline script in `<head>` (BEFORE stylesheet) to prevent FOUC:
    ```html
    <script>
    (function(){var t=localStorage.getItem('theme');if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.setAttribute('data-theme','dark')})();
    </script>
    ```
    **Note**: This is a deliberate exception to the AGENTS.md "no inline scripts" rule — it must run before CSS loads to prevent flash of wrong theme. Keep it as a one-liner.
  - Add `<script src="{{ url_for('static', path='/js/theme.js') }}"></script>` in the `{% block scripts %}` of base.html (before child block scripts)

  **C) New JS file** (`app/static/js/theme.js`):
  - Get toggle button element
  - On click: toggle `data-theme` attribute between absent and `"dark"` on `<html>`
  - Save preference to `localStorage.setItem('theme', 'dark'|'light')`
  - On load: read localStorage, fall back to `prefers-color-scheme` media query
  - Listen for OS theme changes via `matchMedia('(prefers-color-scheme: dark)').addEventListener('change', ...)` — only apply if no saved preference

  **Must NOT do**:
  - Do NOT modify `chat.js` at all
  - Do NOT change any chat API logic
  - Do NOT add npm packages or build tools
  - Do NOT use complex animations or JavaScript-driven style changes (CSS transitions handle it)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Frontend feature with HTML+CSS+JS coordination, visual design decisions for dark colors
  - **Skills**: [`frontend-ui-ux`, `playwright`]
    - `frontend-ui-ux`: Dark theme color palette design, responsive toggle placement
    - `playwright`: Visual verification of both light and dark modes

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: None
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `app/static/css/style.css:1-30` — `:root` custom properties block (created by Task 1)
  - `app/static/css/style.css:23-40` — Navbar styles (need `.navbar-inner` flexbox addition)
  - `app/templates/base.html:10-16` — Current navbar HTML structure
  - `app/static/js/chat.js` — Existing JS pattern (vanilla JS, DOM queries, event listeners) — follow same style but DO NOT edit this file

  **API/Type References**:
  - None (frontend-only)

  **Documentation References**:
  - AGENTS.md: "No inline scripts/styles" — acknowledged exception for FOUC prevention
  - AGENTS.md: "JS: vanilla JavaScript only — no build step, no npm, no bundler"

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Toggle button visible and functional in light mode
    Tool: Playwright (playwright skill)
    Preconditions: Dev server running on localhost:8000, localStorage cleared
    Steps:
      1. Navigate to: http://localhost:8000
      2. Wait for: #theme-toggle visible (timeout: 5s)
      3. Assert: html element does NOT have data-theme attribute (or not "dark")
      4. Assert: body background-color computes to rgb(245, 245, 245)
      5. Assert: #theme-toggle is visible and clickable
      6. Screenshot: .sisyphus/evidence/task-2-light-mode-with-toggle.png
    Expected Result: Light mode renders correctly with toggle visible
    Evidence: .sisyphus/evidence/task-2-light-mode-with-toggle.png

  Scenario: Click toggle switches to dark mode
    Tool: Playwright (playwright skill)
    Preconditions: Page loaded in light mode
    Steps:
      1. Click: #theme-toggle
      2. Wait: 500ms (transition time)
      3. Assert: html[data-theme="dark"] exists
      4. Assert: body background-color computes to approximately rgb(26, 26, 46)
      5. Assert: .navbar background-color computes to approximately rgb(22, 33, 62)
      6. Assert: .chat-messages background-color is dark
      7. Assert: .assistant-message .message-content background-color is dark
      8. Assert: #user-input background-color is dark and text color is light
      9. Assert: .footer background-color is dark
      10. Screenshot: .sisyphus/evidence/task-2-dark-mode-active.png
    Expected Result: All elements use dark theme colors
    Evidence: .sisyphus/evidence/task-2-dark-mode-active.png

  Scenario: Click toggle again returns to light mode
    Tool: Playwright (playwright skill)
    Preconditions: Page in dark mode after previous scenario
    Steps:
      1. Click: #theme-toggle
      2. Wait: 500ms
      3. Assert: html element does NOT have data-theme="dark"
      4. Assert: body background-color computes to rgb(245, 245, 245)
      5. Screenshot: .sisyphus/evidence/task-2-toggle-back-to-light.png
    Expected Result: Cleanly returns to original light mode
    Evidence: .sisyphus/evidence/task-2-toggle-back-to-light.png

  Scenario: Theme persists across page reload
    Tool: Playwright (playwright skill)
    Preconditions: Page loaded
    Steps:
      1. Click: #theme-toggle (switch to dark)
      2. Assert: html[data-theme="dark"] exists
      3. Reload page (page.reload())
      4. Wait for: .chat-container visible (timeout: 5s)
      5. Assert: html[data-theme="dark"] STILL exists (no flash of light)
      6. Assert: body background-color is dark
      7. Screenshot: .sisyphus/evidence/task-2-dark-mode-persisted.png
    Expected Result: Dark mode maintained after reload
    Evidence: .sisyphus/evidence/task-2-dark-mode-persisted.png

  Scenario: Restaurant cards readable in dark mode
    Tool: Playwright (playwright skill)
    Preconditions: Page in dark mode, sample restaurants seeded
    Steps:
      1. Ensure dark mode is active
      2. Type in #user-input: "Italian restaurant"
      3. Click: #send-button
      4. Wait for: .restaurant-card visible (timeout: 15s)
      5. Assert: .restaurant-card background-color is dark
      6. Assert: .restaurant-card h4 color is light (NOT #2c3e50 which would be invisible)
      7. Assert: .restaurant-card p text color is readable (light on dark)
      8. Screenshot: .sisyphus/evidence/task-2-restaurant-cards-dark.png
    Expected Result: Restaurant cards fully readable in dark mode
    Evidence: .sisyphus/evidence/task-2-restaurant-cards-dark.png

  Scenario: No FOUC (flash of unstyled content) on dark mode reload
    Tool: Playwright (playwright skill)
    Preconditions: Dark mode saved in localStorage
    Steps:
      1. Set localStorage theme to "dark" via page.evaluate
      2. Navigate to: http://localhost:8000
      3. Immediately screenshot before any JS loads: .sisyphus/evidence/task-2-no-fouc.png
      4. Assert: html[data-theme="dark"] is set from the inline script
    Expected Result: Dark theme applied before page paints
    Evidence: .sisyphus/evidence/task-2-no-fouc.png
  ```

  **Evidence to Capture:**
  - [ ] .sisyphus/evidence/task-2-light-mode-with-toggle.png
  - [ ] .sisyphus/evidence/task-2-dark-mode-active.png
  - [ ] .sisyphus/evidence/task-2-toggle-back-to-light.png
  - [ ] .sisyphus/evidence/task-2-dark-mode-persisted.png
  - [ ] .sisyphus/evidence/task-2-restaurant-cards-dark.png
  - [ ] .sisyphus/evidence/task-2-no-fouc.png

  **Commit**: YES
  - Message: `feat(ui): add dark mode toggle with localStorage persistence`
  - Files: `app/static/css/style.css`, `app/templates/base.html`, `app/static/js/theme.js`
  - Pre-commit: Playwright visual verification of both themes

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `refactor(css): extract hardcoded colors to CSS custom properties` | `app/static/css/style.css` | Playwright screenshot comparison |
| 2 | `feat(ui): add dark mode toggle with localStorage persistence` | `app/static/css/style.css`, `app/templates/base.html`, `app/static/js/theme.js` | Playwright dark/light toggle tests |

---

## Success Criteria

### Verification Commands
```bash
# Start dev server
uvicorn app.main:app --reload --port 8000
# Then run Playwright scenarios above
```

### Final Checklist
- [ ] Light mode looks identical to current design (no visual regression)
- [ ] Dark mode applies to ALL elements (navbar, chat, cards, input, footer)
- [ ] Toggle button works with clear sun/moon icons
- [ ] Theme persists in localStorage across reloads
- [ ] OS preference respected when no saved preference
- [ ] No FOUC on page load
- [ ] Smooth CSS transitions between themes
- [ ] Restaurant cards fully readable in dark mode (heading color overridden)
- [ ] Links have proper color in both themes
- [ ] No changes to backend or chat functionality
