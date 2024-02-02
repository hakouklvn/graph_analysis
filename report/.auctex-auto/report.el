(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("algorithm2e" "linesnumbered" "ruled" "vlined")))
   (TeX-run-style-hooks
    "latex2e"
    "rep10"
    "algorithm2e"
    "multirow"
    "rotating"
    "pdfpages"
    "graphicx"
    "tikz"
    "preview"))
 :latex)

