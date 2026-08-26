Setup steps

  1. TeX distribution — already installed (TeX Live 2023, confirmed via pdflatex --version). Nothing to do.

  2. VS Code extension — already installed for you: James-Yu.latex-workshop (v10.18.0).

  3. Build config — already added at .vscode/settings.json in this repo, so builds use plain pdflatex (twice, to resolve the table of contents) since
  latexmk isn't on this machine:
  {
    "latex-workshop.latex.tools": [
      { "name": "pdflatex", "command": "pdflatex",
        "args": ["-synctex=1", "-interaction=nonstopmode", "-file-line-error", "%DOC%"] }
    ],  
    "latex-workshop.latex.recipes": [
      { "name": "pdflatex x2", "tools": ["pdflatex", "pdflatex"] }
    ],
    "latex-workshop.latex.recipe.default": "first",
    "latex-workshop.view.pdf.viewer": "tab"
  }

  4. What you need to do now:
  1. Open the project folder in VS Code (code . from the repo root, or File → Open Folder).
  2. Open docs/final_report_latex/main.tex.
  3. Add the missing university.jpeg logo file into docs/final_report_latex/ (the doc currently fails to compile without it — it's referenced on line 67).
     If you don't have the logo yet, comment out that \includegraphics{university.jpeg} line temporarily.
  4. Build: click the green ▶ TeX icon in the top-right of the editor, or press Ctrl+Alt+B.
  5. View the PDF: Ctrl+Alt+V — opens in a tab beside your source, with SyncTeX (click text ↔ jump to matching spot in the PDF, Ctrl+Click in either
     direction).
  6. Check for errors: LaTeX Workshop shows an "PDF LaTeX" output panel and a Problems-tab-style error list if the build fails.
  
  Optional but recommended: install latexmk for better incremental builds:
  sudo apt-get install -y latexmk
  (needs your password — run it yourself in a terminal, or type ! sudo apt-get install -y latexmk here). After that, you can switch
  "latex-workshop.latex.recipe.default" to "latexmk" and add a latexmk-based recipe if you want, though the current pdflatex x2 recipe works fine as-is.

  That's everything needed — the extension, build recipe, and your .tex file are all in place; the only blocker is the missing image.