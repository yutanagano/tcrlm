# tcrlm

This is a codebase for developing and training TCR language models.
This codebase contains code specific to the act of model training.

## Prescribed TCR data format (csv)

All TCR data concerned with this project should be saved as a csv file with
the following columns, in this order, with the specified data:

| Column name | Column datatype | Column contents |
|---|---|---|
|TRAV|`str`|IMGT symbol for the alpha chain V gene (excluding any allele specifiers)|
|CDR3A|`str`|Amino acid sequence of the alpha chain CDR3, including the first C and last W/F residues, in all caps|
|TRAJ|`str`|IMGT symbol for the alpha chain J gene (excluding any allele specifiers)|
|TRBV|`str`|IMGT symbol for the beta chain V gene (excluding any allele specifiers)|
|CDR3B|`str`|Amino acid sequence of the beta chain CDR3, including the first C and last W/F residues, in all caps|
|TRBJ|`str`|IMGT symbol for the beta chain J gene (excluding any allele specifiers)|
|Epitope|`str`|Amino acid sequence of the target epitope, in all caps|
|MHCA|`str`|IMGT symbol for the MHC alpha chain gene (excluding any allele specifiers)|
|MHCB|`str`|IMGT symbol for the MHC beta chain gene (excluding any allele specifiers) (N.B. if class I MHC, the value here is B2M)|
|clone_count|`int`|The number of times this particular TCR-pMHC combination is seen in the dataset|
