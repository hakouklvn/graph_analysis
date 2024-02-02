(TeX-add-style-hook
 "coverpage"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("report" "a4paper" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("svg" "inkscapelatex=false") ("inputenc" "utf8") ("fontenc" "T1") ("geometry" "top=0.6in" "bottom=0.6in" "right=1in" "left=1in")))
   (TeX-run-style-hooks
    "latex2e"
    "report"
    "rep12"
    "svg"
    "inputenc"
    "fontenc"
    "babel"
    "xcolor"
    "graphicx"
    "geometry"))
 :latex)

