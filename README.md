# Setup (Windows)
Download the [release](https://drive.google.com/file/d/1iW07K222Hj5jRN7zWCu1FPaAaRtXIoUk/view?usp=sharing).

Download any desired extra models from [this repository](https://huggingface.co/therealvul/StyleTTS2_GUI_models/tree/main) and unzip them to the Models folder. Below is the expected file structure:
```
Models/
    Multi0_40_24k/
        config.yml
        styles.json
        epoch_2nd_40_1c872.pth
```
`styles.json` is an optional json containing tags and precomputed style vectors. If not supplied the user will have to supply their own reference audio.

Run `styletts2.exe`.

# Usage
![](https://raw.githubusercontent.com/effusiveperiscope/StyleTTS2_GUI/main/docs/1.png)

1. Select model and style at the top.
2. Enter text in the middle.
3. Press the inference button at the bottom to infer.
4. Check tooltips for further guidance on inference settings.

- Sentences not terminated with punctuation can have undesired results.
- ARPABET escapes `{AH0}` and IPA escapes `<dɹˈɪftɪŋ>` are supported. 
- The pipe symbol `|` can be used to condition style diffusion on a different text input: `Hi, my name is Twilight Sparkle.|Oh no! Oh no!`
- You can change the number of generations in `config.yaml` under `n_infer`
- You can drag and drop reference audio into the acoustic and prosodic custom reference file buttons.
- Results are outputted to the `results` directory. You can also drag and drop results out from the corresponding preview button.

# Performance
There is an initial startup cost for the first line synthesized. After that gens are faster.

By default this runs on cpu only. If you have CUDA installed you can try switching the inference device in `config.yaml` to `'cuda'`. On my machine, a line that took ~0.8s/it on CPU took ~0.4s/it on GPU, disregarding the first time synthesis was run.

# Precomputed style vectors
Style vectors are precomputed using `precompute_style_vectors.ipynb`. This relies on a custom indexing schema for the Pony Preservation Project dataset specifically, but it is quite simple so it should be retoolable for other datasets.

# Troubleshooting
- `DirectShowPlayerService::doRender: Unknown error 0x80040266`: On Windows, Qt uses DirectShow which depends on external audio codecs for FLAC playback. Try installing [K-Lite Codec Pack](https://codecguide.com/download_kl.htm). If this becomes enough of an issue I might consider a different audio playback backend.