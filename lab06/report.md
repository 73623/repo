Lab 6: Training Optimization and Experiment Management

The goal of this lab was to investigate practical techniques for training and managing a CNN efficiently. The Fashion-MNIST dataset was used with the same CNN architecture as in Lab 3: two convolutional blocks, each consisting of convolution, ReLU, and max pooling, followed by a flattened linear classifier. The experiments examined MLflow experiment tracking, profiler results, data-loading workers, batch size, learning-rate schedulers, gradient accumulation, and checkpoint recovery.

The environment used a fixed random seed of 42 and ran in the available CPU or GPU environment without changing the model architecture. Training used the Adam optimizer with a learning rate of 0.001 for the YAML-driven baseline. The configured batch size was 128 and the baseline was trained for five epochs.

The baseline CNN was logged with MLflow, including the model configuration, training parameters, per-epoch training and validation metrics, test accuracy, configuration file, and checkpoint. The baseline reached a test accuracy of 89.52% in the recorded MLflow runs. During the five recorded epochs, training loss decreased from approximately 0.535 to 0.263, while validation accuracy increased from 86.48% to 89.74%.

The profiler study measured one epoch and divided the execution time into data loading, forward propagation, backward propagation, and optimizer update. Data loading was the dominant stage and took 13.76 s, or 75.65% of the measured time. The forward pass took 1.85 s (10.15%), backward propagation took 2.03 s (11.14%), and the optimizer step took 0.56 s (3.06%). This result shows that, in this run, computation was not the primary bottleneck: transferring and preparing batches accounted for most of the epoch time.

The number of DataLoader workers was then varied from 0 to 8. The median loading time was 18.36 s for zero workers and 6.69 s for one worker, corresponding to the lowest tested loading time and highest throughput of 8225 samples/s. Two workers took 8.21 s and produced 6702 samples/s, while four and eight workers gave progressively slower results of 13.72 s and 25.48 s. Therefore, one worker was the best setting for this environment. Additional workers introduced overhead and did not improve throughput.

The batch-size study trained for five epochs with batch sizes of 32, 64, 128, 256, and 512. Larger batches generally improved throughput but also increased memory use and showed a slight reduction in test accuracy.

| Batch size | Throughput (samples/s) | Estimated peak memory (MB) | Test accuracy |
|---|---:|---:|---:|
| 32 | 778.5 | 2.53 | 90.42% |
| 64 | 755.8 | 4.83 | 89.49% |
| 128 | 1166.6 | 9.42 | 89.07% |
| 256 | 1456.4 | 18.61 | 89.01% |
| 512 | 1647.8 | 36.98 | 87.54% |

Batch 32 gave the highest test accuracy, while batch 512 gave the highest throughput. The throughput benefit had diminishing and accuracy trade-offs: compared with batch 32, batch 512 reached 2.12 times the throughput but used more memory and lost 2.88 percentage points of test accuracy. Batch 128 provides a reasonable compromise because it combines substantially better throughput than small batches with moderate memory use.

The scheduler and optimizer study trained for ten epochs with Adam and StepLR, Adam and CosineAnnealingLR, Adam and ReduceLROnPlateau, and SGD with momentum and cosine annealing. Adam with ReduceLROnPlateau gave the best test accuracy at 90.61%, closely followed by SGD with momentum and cosine annealing at 90.38%, Adam with cosine annealing at 90.37%, and Adam with StepLR at 90.27%. The differences were small, but the adaptive learning-rate response of ReduceLROnPlateau produced the strongest final result in this comparison.

Gradient accumulation was tested by comparing a true batch of 256 with four accumulated micro-batches of 64. The two approaches used the same effective batch size; for the accumulated implementation, each micro-batch loss was scaled by one quarter before backward propagation so that the accumulated gradients matched the average gradient of a batch of 256. This method allows a large effective batch size to be simulated when memory is limited, although it does not eliminate the added memory needed to retain intermediate computation graphs.

The checkpoint experiment stopped training after epoch 3, saved the model state, optimizer state, PyTorch RNG state, epoch number, and loss history, and resumed through epoch 6. The saved and resumed loss curve continued smoothly beyond the interruption. This verified that restoring both the model and optimizer state, together with the random state, is important for reproducing the training process rather than only restoring the network parameters.

Overall, the results demonstrate that MLflow makes training runs and checkpoints reproducible and inspectable, and that profiling reveals which parts of a pipeline limit performance. Data loading was the largest profiler cost in this environment. One worker provided the best worker scaling, with one worker being faster than both zero and higher worker counts. Larger batches increased throughput and memory use but gradually reduced accuracy. ReduceLROnPlateau was the best scheduler-optimizer combination tested, and gradient accumulation successfully demonstrated an effective-batch-size technique. A checkpoint containing model and optimizer state allowed training to resume continuously after an interruption.