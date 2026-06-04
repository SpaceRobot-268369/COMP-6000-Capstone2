## Goal

Investigate whether AudioGen LoRA can learn species-specific bird vocalisations from BirdNET-detected events.

## Species

* Southern Boobook
* Australian Raven

## Dataset Preparation

BirdNET detections were filtered using confidence thresholds.

Southern Boobook:

* confidence >= 0.9
* 150 event-centred clips

Australian Raven:

* confidence >= 0.7
* 80 event-centred clips

To reduce background contamination, event-centred segmentation was applied around detected vocalisation events.

Several dataset preparation strategies were explored:

* species-specific training sets
* shorter clip durations
* event-centred extraction
* removal of long ambient-only sections

Clip duration was reduced from 8 seconds to 4 seconds to increase the proportion of target vocalisation content.

## Training

Base model:

* facebook/audiogen-medium

Method:

* AudioGen LoRA fine-tuning

Training configuration:

* Epochs: 3
* Learning rate: 5e-5
* Batch size: 1

Separate LoRA adapters were trained for each species.

## Evaluation

Generated audio was manually reviewed using:

* waveform inspection
* spectrogram inspection
* listening tests
* comparison against source recordings

Multiple prompts were tested for each trained adapter.

## Result

Training completed successfully and LoRA adapters were produced for both species.

However, generated outputs frequently reproduced ambient environmental sounds rather than species-specific vocalisations.

Southern Boobook generation rarely produced recognisable owl calls.

Australian Raven generation occasionally produced call-like sounds but remained inconsistent and unreliable.

The attempt did not achieve the desired quality threshold for species-specific bird sound generation.

## Conclusion

AudioGen LoRA was able to learn general acoustic characteristics of the recordings but struggled to reliably capture distinctive bird vocalisation patterns.

Further work is required to improve dataset quality and isolate vocalisation events more effectively.
