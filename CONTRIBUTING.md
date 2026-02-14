# Contributing to EY Challenge 2026

Thank you for your interest in contributing! We follow a standard **Fork & Pull Request** workflow to maintain code quality.

## 🌏 Workflow Overview / 工作流概览

1.  **Fork** the repository to your own account (e.g., `yuanweize/EY_Challenge_2026`).
2.  **Clone** your fork locally:
    ```bash
    git clone https://github.com/yuanweize/EY_Challenge_2026.git
    ```
3.  **Add Upstream** remote to keep in sync:
    ```bash
    git remote add upstream https://github.com/eurun/EY_Challenge_2026.git
    ```
4.  **Create a Branch** for your feature/fix:
    ```bash
    git checkout -b feature/my-new-model
    ```
5.  **Commit** your changes (Atomic commits preferred).
6.  **Push** to your fork:
    ```bash
    git push origin feature/my-new-model
    ```
7.  **Open a Pull Request** (PR) against the `eurun/master` branch.

## 📝 Coding Standards / 代码规范

-   **Notebooks**: Clean output cells before committing to reduce diff noise (unless output is necessary for demonstration).
-   **Documentation**: Maintain the **Bilingual (English/Chinese)** format in LaTeX files.
-   **Commits**: Use clear messages (e.g., `feat: add landsat extraction logic`, `fix: correct typo in rules`).

## 🐛 Reporting Issues

Please use the [Issue Templates](.github/ISSUE_TEMPLATE/) to report bugs or request features.
