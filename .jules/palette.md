## 2025-04-24 - Add accessibility attributes to controls in App.js
**Learning:** A recurring UI/accessibility pattern in this frontend codebase is the omission of proper form field associations and missing ARIA attributes for complex controls, such as grouped buttons.
**Action:** When working on form inputs and controls, actively look for and apply `htmlFor`/`id` linking for standard inputs, and use ARIA attributes like `role="group"`, `aria-labelledby`, and `aria-pressed` for custom controls or button groups.
