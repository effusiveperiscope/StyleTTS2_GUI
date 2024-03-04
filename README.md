# Setup (Windows)
Download the release.

Download models from this repository and unzip them to the Models folder.
```
Models/
    Multi0_40_24k/
        config.yml
        styles.json
        epoch_2nd_40_1c872.pth
```
`styles.json` is an optional json containing tags and precomputed style vectors. If not supplied the user will have to supply their own reference audio.

Run `.exe`.

# Usage tips
- Sentences not terminated with punctuation can have undesired results.
- ARPABET escapes `{AH0}` and IPA escapes `<>` are supported. 
- The pipe symbol `|` can be used to condition style diffusion on a different text input: `Hi, my name is Twilight Sparkle.|Oh no! Oh no!`

# Performance
There is an initial startup cost for the first line synthesized. After that gens are faster.

By default this runs on cpu only. If you have CUDA installed you can try switching the inference device in `config.yaml` to `'cuda'`. On my machine, a line that took ~0.8s/it on CPU took ~0.4s/it on GPU, disregarding the first time synthesis was run.