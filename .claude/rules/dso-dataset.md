# DSO dataset

- 24 train classes (things 0–8, stuff 9–23), ignore 24; the head is 25-way. The raw→train mapping is `seg_label_mapping` in `configs/_base_/datasets/dso_panoptic_lpmix.py`, and any unlisted raw id maps to ignore.
- Raw 17 (Stop) and 28 (Others) are ignored in training; they are the OOD classes.
- `dso_infos_test_cetran.pkl` is exactly the held-out test split plus the Cetran sequences (same frames, same order).
- `data/dso/dso_infos_mini.pkl` on disk is 5 held-out-test frames containing Stop/Others, used for OOD smoke tests. `create_data.py dso` overwrites it with the converter's own mini split, 32 frames of a train sequence (`MINI_SEQUENCE` in `tools/dataset_converters/dso_converter.py`).
