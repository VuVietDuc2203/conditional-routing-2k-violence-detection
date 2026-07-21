# Third-party data sources and reconstruction boundary

Raw third-party videos are not redistributed in this artifact. The retained
cohort can be audited through `data/manifests/corpus_lineage.csv` and
`data/manifests/retained_source_reconstruction.csv`. Internal dataset keys map
to public sources as follows.

| Internal key | Source collection | Source/citation route | Release note |
|---|---|---|---|
| `hockey` | Hockey Fights | Paper DOI: https://doi.org/10.1007/978-3-642-23678-5_39 | Obtain from the dataset authors or an authorized mirror; no raw files are included here. |
| `movies2` | Real-Life Violence Situations | https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset; paper DOI: https://doi.org/10.1109/ICICIS46948.2019.9014714 | The internal key is historical and does not mean the 200-clip Movies Fight dataset. Data files remain credited to the original authors. |
| `rwf2000` | RWF-2000 subset | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection; paper DOI: https://doi.org/10.1109/ICPR48806.2021.9412502 | The official repository states that the database may not be modified or redistributed without SMIIP Lab approval and that video files are not currently downloadable there. This artifact therefore contains only derived IDs, split metadata, predictions, and hashes. |
| `surv_fight` | Surveillance Camera Fight Dataset | https://github.com/seymanurakti/fight-detection-surv-dataset; paper DOI: https://doi.org/10.1109/IPTA.2019.8936070 | The source repository publishes the 300-video dataset and an MIT license; this artifact still omits the media to keep one consistent third-party-data boundary. |
| `violent_flows` | Violent-Flows Crowd Violence/Non-violence Database | https://talhassner.github.io/home/projects/violentflows/index.html; paper DOI: https://doi.org/10.1109/CVPRW.2012.6239348 | The source page requires a request form for download credentials. |

## Reconstruction procedure

1. Acquire each source collection directly from its owner under the applicable
   terms. Do not use this package as a grant of media rights.
2. Use `original_dataset_relative` in
   `retained_source_reconstruction.csv` to map the author's retained items to
   the acquired source tree. The CSV contains no absolute host paths.
3. Use `video_id`, `label`, `split`, and `semantic_group_id` from the manifest;
   do not create a new random split.
4. Verify `raw_video_sha256` against the acquired source byte before mapping it
   into the retained cohort. All 3,516 rows have a raw-video hash; 3,497 hashes
   are unique, with 19 duplicate-content groups preserved under distinct IDs.
   The separate `m1_cached_tensor_sha256` field authenticates the frozen
   whole-frame T50 cache and is not a raw-video content hash.
5. Reproduce source-wise results from the distributed frozen prediction files;
   no source video is required for that aggregation.

The matched replay media were deterministically composed with seed 50900 using
the exact builder in `code/validation_code/build_streaming_workloads.py` and
the earlier cache manifest preserved as
`data/workloads/builder_input_manifest.csv`. Sanitized segment selections,
encoded timeline sidecars, and the mapping to the later semantic-group split
are distributed in `data/workloads/`; the controlled workload labels were
unavailable to the runtime benchmark. Reconstructing the three MP4 files still
requires lawful access to the source clips listed above.

The source archive supplied 350 inputs carrying the `rwf2000` key; 181 remained
in the frozen train/validation/test cohort after the common exclusions. This is
not a claim that all 2,000 RWF-2000 videos were evaluated.
