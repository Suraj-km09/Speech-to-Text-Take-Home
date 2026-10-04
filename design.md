# Speech-to-Text Take-Home: Design Notes

**Google Colab Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1kJUehF-SwyXdeiZL673BAvJUEhc-YQpo?usp=sharing)

## Goal

I built a small speech-to-text pipeline in PyTorch, starting from raw audio and ending at decoded text, and trained it to overfit 15 audio clips. I wasn't aiming for real-world accuracy here. What I wanted to get right was the plumbing: variable-length batching, downsampling, CTC input lengths, and decoding.

## Data and features

I picked 15 utterances from LibriSpeech `dev-clean`. Audio is 16 kHz, and I turn it into 80-bin Mel-spectrograms with an FFT size of 400 and a hop length of 160, so each frame covers 10 ms of audio. Features are laid out as `[mel, time]`.

Clips differ in length, so I wrote a `collate_fn` that pads each batch to its longest item. It also returns the original `mel_lengths`, because CTC needs the true length of each input and not the padded one.

## Tokenizer

I went with characters rather than words. The vocabulary is `a-z`, space, apostrophe, and a blank token `_` for CTC, which gets its own index. Transcripts are mapped to integer ids and flattened into one long tensor before going into `CTCLoss`, along with a separate tensor of target lengths.

## Model

The network is two Conv1D layers, then a 2-layer bidirectional LSTM, then a linear layer over the vocabulary.

```text
mel-spectrogram
  -> Conv1D(128, k=5, stride=2) + BatchNorm + ReLU
  -> Conv1D(128, k=5, stride=2) + BatchNorm + ReLU
  -> BiLSTM (2 layers, hidden size 128)
  -> Linear
  -> character logits
```

## Downsampling

Both convolutions use kernel 5, stride 2 and padding 2, so each one roughly halves the time axis and the pair gives about 4x. That shortens the sequence the LSTM has to go through, while still leaving plenty of frames for CTC to align characters to.

I don't assume the output is exactly `T / 4`. After each conv I compute the length with the usual formula:

```text
out = floor((in + 2*padding - kernel) / stride) + 1
```

and apply it twice to the original mel lengths. Those numbers are what get passed to CTC as the input lengths.

## Training

Each step goes like this:

1. The model outputs logits of shape `[B, T, V]`.
2. I apply `log_softmax` over the vocab dimension.
3. I transpose to `[T, B, V]`, which is what `CTCLoss` expects.
4. Input lengths come from the conv length formula above.
5. Targets are flattened and passed with their per-sample lengths.

CTC takes care of the alignment between frames and characters, so no frame-level labels are needed. I used Adam with a learning rate of 1e-3 and ran enough epochs for the model to memorize the 15 clips.

## Decoding

For decoding I use plain greedy CTC. Take the argmax at every frame (only up to that sample's valid length), collapse repeated tokens, then drop the blanks. What's left is the predicted text.

## Results

I ran the trained model on the same 15 clips and measured exact match, character error rate, character accuracy, and edit distance.

```text
samples          : 15
exact match      : 100.00%
average CER      : 0.0000
character acc    : 100.00%
```

Every transcript came out identical to the ground truth. That's the expected outcome for a model trained on only these 15 clips, and it mostly tells me the pipeline is wired up correctly. It says nothing about how well the model would do on unseen speech. Training logs, the loss curve, sample predictions and the full results table are in the notebook.
