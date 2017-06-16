1. Download and extract the corpus for "http://nactem.ac.uk/MLEE/" site.
2. Download the pretrained word embeddings with file name "PubMed-w2v.bin" form "http://evexdb.org/pmresources/vec-space-models/" website.
3. Update the path of word embeddings file in loadWordEmbeddings method in Utils.py(for RNN and FFNN models) file and in Trigger.py(for CNN model).
4. Run PrePreprocess.sh file to create the folder structure.
5. Copy all the corpus files into Corpus folder.
6. Copy the file form ReplaceFile folder and replace it with the file in "./standoff/test/train/" folder (So as to correct an annotation mistake)
7. Run Preprocess.sh file to perform preprocessing.
8. Run Trigger.py file to train and test the model.
9. Run Process3.py file to get the final F1-scores and the analysis results.
