# Scripts

Steps required for REFD approach (more info [here](../README.md)):
* `preprocess_dataset.py`,
* `ood_detector.py`,
* `get_classification_metrics.py`.


The directory contains the following auxiliary scripts:
* `download_resources.py` - required to download Special Sessio dataset,
* `generate_embeddings.py` - generate embeddings for each subset,


## Generate embeddings

Use your embeddings architecture to generate embeddings for train, dev and test subsets.

```bash
python scripts/generate_embeddings.py --config $config_path --embeddings_root_dir $output_emb_dir
```

The generated structure looks like this:
```
$output_emb_dir
├── dev
│   ├── 0 # class name mapped to an ID
│   │   ├── filename_1.npy # embedding of the filename_1 audio file
│   │   ├── ...
│   │   └── filename_n.npy
│   ├── ...
│   └── 1
│       ├── filename_1.npy
│       ├── ...
│       └── filename_n.npy
├── test
│   ├── 0
│   │   ├── filename_1.npy
│   │   ├── ...
│   │   └── filename_n.npy
│   ├── ...
│   └── 1
│       ├── filename_1.npy
│       ├── ...
│       └── filename_n.npy
└── train
    ├── 0
    │   ├── filename_1.npy
    │   ├── ...
    │   └── filename_n.npy
    ├── ...
    └── 1
        ├── filename_1.npy
        ├── ...
        └── filename_n.npy
```

## Visualize embeddings

To visualize embeddings created using `scripts/generate_embeddings.py` script use `notebooks/VisualizeEmbeddings.ipynb`. The notebook creates UMAP visualization of train and test subsets.


