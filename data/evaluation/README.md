# Evaluation Data Placeholder

Place evaluation JSON files here:

```text
data/evaluation/
+-- eval_nextgqa_mixed_server.json
+-- eval_star_mixed_server.json
```

Suggested video folders:

```text
data/NExTQA/videos/
data/STAR/
```

The evaluation code reads the video path from each sample's `video` field. Make sure those paths point to your local NExT-QA/NExT-GQA and STAR video files.
