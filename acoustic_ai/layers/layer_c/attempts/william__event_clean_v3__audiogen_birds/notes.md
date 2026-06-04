\## Initial Observation



Generated outputs contained substantial background ambience.



Many generated samples resembled environmental recordings rather than target bird calls.



Southern Boobook outputs rarely produced recognisable owl vocalisations.



Australian Raven outputs occasionally produced call-like sounds but remained inconsistent.



\## Attempts



\### Attempt 1 – High Confidence BirdNET Events



Dataset filtered using BirdNET confidence thresholds.



Result:



\* Reduced obvious false detections.

\* Background noise remained dominant.



\### Attempt 2 – Species-Specific Datasets



Created independent datasets for Southern Boobook and Australian Raven.



Result:



\* Reduced class mixing.

\* No significant improvement in generation quality.



\### Attempt 3 – Event-Centred Segmentation



Clips were centred on BirdNET-detected events.



Result:



\* Improved localisation of bird activity.

\* Model continued learning ambient acoustic context.



\### Attempt 4 – Shorter Clip Duration



Reduced clip length from 8 seconds to 4 seconds.



Result:



\* Reduced unrelated environmental content.

\* Bird vocalisations still not learned reliably.



\### Attempt 5 – Multiple LoRA Training Runs



Repeated training with different datasets and segmentation strategies.



Result:



\* Training consistently converged.

\* Generated outputs remained dominated by ambient sound.



\## Failure Analysis



The primary failure mode appears to be insufficient isolation of target vocalisations.



Although BirdNET detections identify the presence of a species, the surrounding audio often contains:



\* wind

\* insects

\* frogs

\* environmental noise

\* other birds



AudioGen appears to learn dominant acoustic patterns from the training clips, causing background ambience to be reproduced more consistently than the target vocalisations.



This effect was particularly visible for Southern Boobook, where generated outputs frequently lacked recognisable owl calls.



\## Possible Future Work



Potential improvements include:



\* spectral denoising

\* frequency-based vocalisation extraction

\* CNN-based vocalisation isolation

\* BirdNET-guided call extraction

\* stronger event segmentation

\* cleaner open-source bird-call datasets

\* manual validation of training clips before LoRA training



