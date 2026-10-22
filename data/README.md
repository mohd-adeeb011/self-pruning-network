# UCI Covertype data

`covertype.zip` is the original archive from the [UCI Covertype
page](https://archive.ics.uci.edu/dataset/31/covertype). It contains
`covtype.data.gz`: 581,012 rows, 54 inputs, and one class label per row.
The loader checks the decompressed data against SHA-256
`0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945`.

Citation: Blackard, J. (1998). *Covertype* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C50K5N

UCI lists a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)
for this dataset. The archive is included so this repository runs without a
download. The loader uses the original rows without changing their labels.
