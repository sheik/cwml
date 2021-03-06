model:
  patience: 2                # Patience setting for keras early stopping
  epochs: 50                 # Number of epochs 
  resize:                    # How to resize image
    x: {{ variables['resize_x'] }} 
    y: {{ variables['resize_y'] }}   
data:
  corpus: "corpus.txt"
  jitter: {{ variables['jitter'] }}                # Jitter is a number between 0.0 and 1.0 that determines how sloppy to "key" the samples 
  samples_per_phrase: {{ variables['samples_per_phrase'] }}    # Number of samples to generate per phrase. Samples will range in wpm, jitter, and noise
  max_phrases: {{ variables['max_phrases'] }}           # Number of phrases to take from the corpus. Specify 0 to take all 
  sample_rate: 8000          # Sample rate for the generated data
  sample_length: {{ variables['sample_length'] }}        # Approximate length samples should be in seconds
  remove_stopwords: {{ variables['remove_stopwords'] }}    # Remove English stop words
  wpm_range:                 # Range of WPM for the generated data
    low: 20
    high: 25
  snr_range:                 # Range of SNR for the generated data
    low: {{ variables['snr_low'] }} 
    high: {{ variables['snr_high'] }} 
system:
  jobs: 12                     # Number of processes to run in parallel when generating data
  gpu_enabled: True            # Is a GPU being used on this system?
  multi_gpu_enabled: False     # Should a GPU fabric be used (new/unstable)
  volumes:
    data: "/mnt/raid/{{ variables['data_dir'] }}"             # Location to put generated wav files (training data)
    model: "/mnt/raid/{{ variables['model_dir'] }}"         # Location to save model 
    test: "/mnt/raid/{{ variables['test_dir'] }}"     # Location to save model test files 

