
DOC_DIR="doc"
TMP_DIR="tmp"
PDFLATEX="pdflatex -output-directory " + TMP_DIR + " -halt-on-error -interaction=nonstopmode "

rule runLatex:
    output: 
        TMP_DIR + "/{name}.pdf"
    input: 
        DOC_DIR + "/{name}.tex"
    shell: 
        PDFLATEX + "{input}"
     
rule doc: 
    input:
        TMP_DIR + "/overview.pdf"
