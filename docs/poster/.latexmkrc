# Tell latexmk to always use lualatex for PDF generation
$pdf_mode = 4;
$postscript_mode = $dvi_mode = 0;

# Explicitly define the lualatex call commands
$lualatex = 'lualatex -file-line-error -synctex=1 -interaction=nonstopmode %O %S';
