# sad-model

This is my pet project which aims to create a SAD model from scratch.

What did i do:
1) Wrote a custom class that converts a raw audio signal into a log mel-spectrogram using PyTorch so that it could leverage GPUs.
2) Processed an exeisting dataset, took a sample from it, as i don't a lot of compute (everything was on Google Colab's GPUs and the environment would regularly crash when attempitng to load and process the entire dataset), and converted it into a log-mel spectrogram using my custom class.
3) Succesfully rained a GRU based model to detect speech activity.

Here are the metrics on the test set:
| Metric    | Value                  |
| --------- | ---------------------- |
| Accuracy  | 0.9998331655911613     |
| Hamming   | 0.00016682081819592185 |
| Precision | 0.9327198181417427     |
| Recall    | 0.9306135245038709     |
| F1        | 0.9296357635399213     |
| Loss      | 0.0008604296028513627  |

Here's the link to processed dataset - https://huggingface.co/datasets/hypersunflower/ava_speech_data_log_mel_spec

Link to the original dataset - https://huggingface.co/datasets/nccratliri/vad-human-ava-speech

Link to my model - https://huggingface.co/hypersunflower/a_sad_model

More detailed explanation of repo:
* logMelSpectrogram.py - contains the custom class for log mel-spectrogram
* sadModel.py - model's architecture
* speech_detection.py - a class that handles inference
* trainerAndTester.py - class that contains train and test loops, i've tried to make it as applicable to other use cases as possible
* model_training.ipynb - notebook that contains model traing code
* data_processing.ipynb - notebook that contains code with which a processed the data
* example_inference.ipynb - notebook has, as the name implies, usage example


Note: even though the model demostrates a rather good performances on the test set of data, it will probably fail in the real world, due to the vad-human-ava-speech consisting of very clean audios that lack external noise.

SAD - Speech Activeity Detection
