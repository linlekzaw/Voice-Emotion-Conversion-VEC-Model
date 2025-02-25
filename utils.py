{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":30918,"isInternetEnabled":true,"language":"python","sourceType":"notebook","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import tensorflow as tf\nimport os\nimport random\nimport numpy as np\nimport librosa\nimport pyworld\n\ndef l1_loss(y, y_hat):\n    return tf.reduce_mean(tf.abs(y - y_hat))\n\ndef l2_loss(y, y_hat):\n    return tf.reduce_mean(tf.square(y - y_hat))\n\ndef cross_entropy_loss(logits, labels):\n    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))\n\ndef load_wavs(wav_dir, sr):\n    wavs = []\n    for file in os.listdir(wav_dir):\n        file_path = os.path.join(wav_dir, file)\n        wav, _ = librosa.load(file_path, sr=sr, mono=True)\n        wavs.append(wav)\n    return wavs\n\ndef world_decompose(wav, fs, frame_period=5.0):\n    wav = wav.astype(np.float64)\n    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)\n    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)\n    ap = pyworld.d4c(wav, f0, timeaxis, fs)\n    return f0, timeaxis, sp, ap\n\ndef world_encode_spectral_envelop(sp, fs, dim=24):\n    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)\n    return coded_sp\n\ndef world_decode_spectral_envelop(coded_sp, fs):\n    fftlen = pyworld.get_cheaptrick_fft_size(fs)\n    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)\n    return decoded_sp\n\ndef extract_emotion_features(wav, sr, n_mfcc=24):\n    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)\n    chroma = librosa.feature.chroma_stft(y=wav, sr=sr)\n    spectral_contrast = librosa.feature.spectral_contrast(y=wav, sr=sr)\n    return np.vstack((mfcc, chroma, spectral_contrast))\n\ndef pitch_shift(wav, sr, shift_steps=2):\n    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=shift_steps)\n\ndef time_stretch(wav, rate=1.1):\n    return librosa.effects.time_stretch(wav, rate=rate)\n\ndef augment_audio(wav, sr):\n    augmented_wavs = []\n    augmented_wavs.append(pitch_shift(wav, sr, shift_steps=2))\n    augmented_wavs.append(pitch_shift(wav, sr, shift_steps=-2))\n    augmented_wavs.append(time_stretch(wav, rate=1.2))\n    augmented_wavs.append(time_stretch(wav, rate=0.8))\n    return augmented_wavs\n\ndef normalize_features(features):\n    mean = np.mean(features, axis=1, keepdims=True)\n    std = np.std(features, axis=1, keepdims=True)\n    return (features - mean) / std, mean, std\n\ndef denormalize_features(normalized_features, mean, std):\n    return (normalized_features * std) + mean\n\ndef convert_emotion(input_features, target_mean, target_std):\n    return (input_features - target_mean) / target_std\n\ndef world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):\n    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)\n    return wav.astype(np.float32)\n\ndef save_audio(file_path, audio, sr=16000):\n    librosa.output.write_wav(file_path, audio, sr)","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","trusted":true,"execution":{"iopub.status.busy":"2025-02-25T04:22:13.157406Z","iopub.execute_input":"2025-02-25T04:22:13.157797Z","iopub.status.idle":"2025-02-25T04:22:13.594695Z","shell.execute_reply.started":"2025-02-25T04:22:13.157764Z","shell.execute_reply":"2025-02-25T04:22:13.593736Z"}},"outputs":[],"execution_count":4},{"cell_type":"code","source":"!pip install pyworld\n","metadata":{"trusted":true,"execution":{"iopub.status.busy":"2025-02-25T04:21:37.870362Z","iopub.execute_input":"2025-02-25T04:21:37.870703Z","iopub.status.idle":"2025-02-25T04:22:07.755966Z","shell.execute_reply.started":"2025-02-25T04:21:37.870673Z","shell.execute_reply":"2025-02-25T04:22:07.755034Z"}},"outputs":[{"name":"stdout","text":"Collecting pyworld\n  Downloading pyworld-0.3.5.tar.gz (261 kB)\n\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m261.0/261.0 kB\u001b[0m \u001b[31m9.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n\u001b[?25h  Installing build dependencies ... \u001b[?25l\u001b[?25hdone\n  Getting requirements to build wheel ... \u001b[?25l\u001b[?25hdone\n  Preparing metadata (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\nRequirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from pyworld) (1.26.4)\nRequirement already satisfied: mkl_fft in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (1.3.8)\nRequirement already satisfied: mkl_random in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (1.2.4)\nRequirement already satisfied: mkl_umath in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (0.1.1)\nRequirement already satisfied: mkl in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (2025.0.1)\nRequirement already satisfied: tbb4py in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (2022.0.0)\nRequirement already satisfied: mkl-service in /usr/local/lib/python3.10/dist-packages (from numpy->pyworld) (2.4.1)\nRequirement already satisfied: intel-openmp>=2024 in /usr/local/lib/python3.10/dist-packages (from mkl->numpy->pyworld) (2024.2.0)\nRequirement already satisfied: tbb==2022.* in /usr/local/lib/python3.10/dist-packages (from mkl->numpy->pyworld) (2022.0.0)\nRequirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.10/dist-packages (from tbb==2022.*->mkl->numpy->pyworld) (1.2.0)\nRequirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.10/dist-packages (from mkl_umath->numpy->pyworld) (2024.2.0)\nRequirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.10/dist-packages (from intel-openmp>=2024->mkl->numpy->pyworld) (2024.2.0)\nBuilding wheels for collected packages: pyworld\n  Building wheel for pyworld (pyproject.toml) ... \u001b[?25l\u001b[?25hdone\n  Created wheel for pyworld: filename=pyworld-0.3.5-cp310-cp310-linux_x86_64.whl size=859237 sha256=b8413883bfce1df662674fe20e61374b02e418e8cd031cf79f347867d1bcdb1d\n  Stored in directory: /root/.cache/pip/wheels/8e/a0/94/52e99161f9460670f11129bff5224ddf1a17915007d8cfa196\nSuccessfully built pyworld\nInstalling collected packages: pyworld\nSuccessfully installed pyworld-0.3.5\n","output_type":"stream"}],"execution_count":3}]}