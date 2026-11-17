#!/bin/sh
# Builds report.pdf from report.tex, sections/ and the generated figures and tables in gen/.
# Needs pdflatex with TikZ, booktabs, tabularx, longtable, placeins, microtype and KOMA-Script.
set -e
cd "$(dirname "$0")"
for n in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error report.tex > build.log 2>&1 || { grep -n "^!" -A 6 build.log | head -n 40; exit 1; }; done
grep -c "Overfull" build.log | sed 's/^/overfull boxes: /'; grep "LaTeX Warning: Reference\|undefined\|multiply" build.log | sort -u | head -n 10
pdfinfo report.pdf | grep Pages
if [ "$1" != "--keep" ]; then rm -f report.aux report.log report.toc report.out build.log; fi
