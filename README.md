# T5 Fine-Tuning for Abstractive Summarization

This notebook demonstrates how to fine-tune the T5 model for the task of abstractive summarization using Hugging Face's Transformers library.

## 🚀 Goals
- Fine-tune the T5 model on a custom summarization dataset.
- Use the trained model to generate summaries for unseen input text.

## 🧰 Requirements
The notebook uses the following Python libraries:

```bash
pip install transformers datasets evaluate rouge_score
```

## 📘 Contents
- **Dataset Loading**: Utilizes the `datasets` library to load and preprocess the summarization dataset.
- **Model and Tokenizer Setup**: Initializes the T5 tokenizer and model for conditional generation.
- **Training**: Fine-tunes the model using the Hugging Face Trainer API.
- **Evaluation**: Uses ROUGE metrics to evaluate the summarization performance.
- **Inference**: Demonstrates how to use the fine-tuned model to summarize new texts.

## 📝 Notes
- Make sure you have a GPU-enabled environment for training.
- The notebook is ideal for experimentation and understanding how transformer-based models like T5 can be customized for summarization tasks.

## 📄 License
This project is open-source and uses permissive licensing (e.g., Apache 2.0) unless otherwise specified in the repository.

Feel free to use and modify this notebook for your own projects!

# 🗣️ ASR Demo using Transformers

This notebook demonstrates how to perform Automatic Speech Recognition (ASR) using Hugging Face's `transformers` and `datasets` libraries.

## 🚀 Goals
- Transcribe audio files into text using a pre-trained ASR model.
- Showcase the complete ASR pipeline from audio input to textual output.

## 🧰 Requirements
The notebook uses the following Python libraries:

```bash
pip install transformers datasets torchaudio librosa soundfile
```
## 📘 Contents
- **Dataset Loading**: Loads or simulates audio input using torchaudio or custom audio samples.
- **Preprocessing**: Handles resampling and normalization of audio input for compatibility with transformer models.
- **Model and Tokenizer Setup**: Loads a pre-trained ASR model (e.g., `facebook/wav2vec2-base-960h`) and its processor.
- **Inference**: Converts speech into text using the model and displays transcriptions.
- **Evaluation** *(optional)*: If reference transcriptions are available, computes metrics such as Word Error Rate (WER).

## 📝 Notes
- A GPU is recommended for faster inference, especially on longer audio samples.
- Ensure that your input audio format is compatible (typically WAV or FLAC, mono, 16kHz).
- Modify the processor/model paths to use other ASR models from Hugging Face’s model hub.

## 📄 License
This project is open-source and uses permissive licensing (e.g., Apache 2.0) unless otherwise specified in the repository.

Feel free to use and modify this notebook for your own speech-to-text projects!


