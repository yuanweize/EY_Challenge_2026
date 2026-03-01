---
name: enforce_latex_documentation
description: Ensures that all thoughts, analysis, footprint, and changes are documented comprehensively into the project's LaTeX documentation.
---

# Enforce LaTeX Documentation (强制记入 LaTeX 文档机制)

This skill dictates a strict procedural requirement for all tasks conducted in this project.

## Rule / 规定
**EVERY** analysis, decision, code structure design, and problem-solving process MUST be recorded in the `doc/chapters/07_development_log.tex` (or other appropriate chapters) of the project's LaTeX documentation.

1. **When to do this**: 
   - After completing an analysis or reaching a conclusion (like environment setup, data exploration).
   - Before moving on to the next major execution step. 
   - Whenever the AI designs a solution (e.g. model architecture, data cleaning pipeline).
   - **Whenever the user provides an Online Evaluation/Leaderboard Score**.
2. **How to do this (BILINGUAL PARITY IS MANDATORY)**:
   - **Crucial Rule**: The document MUST be written in strict English and Simplified Chinese (简体中文) paragraph-by-paragraph translation format.
   - Example formatting:
     `An end-to-end multi-model stacking script was orchestrated.`
     `(一个端到端的多模型集成脚本被成功编排。)`
   - Every single section, subsection, bullet point, and infobox must have its Chinese equivalent situated immediately afterwards or natively integrated. DO NOT WRITE ENGLISH-ONLY BLOCKS.
   - Use proper LaTeX formatting (`\section`, `\subsection`, `\begin{itemize}`, etc.).
3. **Mandatory Evaluation Workflow (Local $\rightarrow$ Online)**:
   - **Step 1**: After every optimization, you MUST first run the local evaluation script (`evaluate_local.py`) and record the local metrics in the document.
   - **Step 2**: You MUST then explicitly prompt the user (via `notify_user`) to submit the generated `.csv` file to the online Leaderboard and return the official score to you.
   - **Step 3**: Once the user provides the Leaderboard score, create a distinct sub-section titled "Online Evaluation Record / 线上评估存档". Record the exact Date/Time, the Official Score, Algorithm details, and a scientific diagnosis comparing the online score with the local metrics to calibrate our local evaluation script.
   - (工作流强制规定：每次优化后，必须先跑本地评估并记录文档。随后明确提示最高指挥官进行线上提交。获得返回的线上得分后，必须记入“线上评估存档”以用于校准本地跑分脚本。)
4. **Compilation**:
   - After editing the `.tex` files, ALWAYS compile the document to ensure it builds correctly. The project uses a `Makefile` in the `doc/` directory. Run `cd doc && make` to build the new PDF into `doc/dist/`.

**Never skip the English-Chinese bilingual translation.** It is a core requirement of this repository.
