## 2025-05-18 - Missing Form Field Associations Pattern
**Learning:** This codebase had a widespread pattern of form controls missing proper associations in `frontend/src/App.js` (e.g., `<label>` without `htmlFor` and `<input>` without `id`). Side-by-side inputs (like Rows x Cols) lacked grouped descriptions.
**Action:** Always check newly modified or created forms for explicit `htmlFor`/`id` linking. For adjacent input pairings sharing a label, wrap them in `role="group"` with `aria-labelledby` and provide individual `aria-label`s to each input.
