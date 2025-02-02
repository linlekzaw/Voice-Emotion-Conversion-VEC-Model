🗣️ #Voice Emotion Conversion (VEC) Model


Convert neutral speech to expressive emotions using deep learning



📌 ##Overview


This project aims to transform the emotion in an audio sample while preserving the speaker's identity. Unlike emotion recognition models, this approach modifies the emotional tone of speech using deep learning techniques like CycleGAN-VC, StarGAN-VC, and HiFi-GAN.

**🔥 Features


✅ Converts neutral speech to emotions like happy, sad, angry, surprised


✅ Uses CycleGAN-VC for non-parallel emotion conversion


✅ Generates high-quality speech with HiFi-GAN vocoder


✅ Pre-trained emotion embeddings for faster inference


**📂 Dataset


This model is trained on publicly available emotional speech datasets:

IEMOCAP
CREMA-D
RAVDESS


**🏗️ Model Architecture


1️⃣ Feature Extraction
Extract Mel spectrograms, F0 (pitch), and energy from input speech.
Normalize features across the dataset.
2️⃣ Emotion Conversion (CycleGAN-VC / StarGAN-VC)
Train a CycleGAN model to convert emotions while preserving voice identity.
Use StarGAN-VC2 for multi-emotion conversion.
3️⃣ Speech Synthesis (HiFi-GAN)
Convert the modified spectrogram back to a realistic waveform.
