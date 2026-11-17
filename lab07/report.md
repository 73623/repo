Lab 7: Model Compression and Knowledge Distillation

 Objective

This lab compared model pruning and knowledge distillation as ways to reduce the size and inference cost of CNNs while retaining Fashion-MNIST classification accuracy. The saved results include test accuracy across pruning levels and measurements of parameter count, checkpoint size, and CPU inference latency for selected models.

 Baseline

The baseline CNN contains three convolutional layers (32, 64, and 128 output channels) and fully connected layers of 768 and 128 units. It was trained for six epochs using Adam and cross-entropy loss. It achieved 91.45% test accuracy, with 1,077,898 parameters, a 4.116 MB checkpoint, and a measured single-sample CPU latency of 0.800 ms.

Pruning results

Unstructured global L1 pruning removed the lowest-magnitude weights across convolutional and linear layers. Each pruned model was fine-tuned for three epochs. Accuracy first improved as sparsity increased, peaked at 50–75%, and then declined at higher sparsity:

| Unstructured sparsity | Test accuracy |
|---:|---:|
| 0% (baseline) | 91.45% |
| 25% | 91.83% |
| 50% | 92.32% |
| 75% | 92.32% |
| 90% | 92.21% |
| 95% | 91.88% |

At 90% sparsity, accuracy was 92.21%. Saving this model as a dense checkpoint used 4.116 MB, the same size as the baseline; saving the weights in sparse tensor format used 2.467 MB. CPU latency was 0.805 ms, essentially unchanged from the baseline. Thus, although the sparse-format artifact was smaller, the measured dense inference did not become faster.

Structured L2 filter pruning was applied to the convolution layers and followed by three epochs of fine-tuning. The accuracy results were 91.47% after 10% filter removal, 91.98% after 25%, and 91.30% after 50%. The 50% result is 0.15 percentage points below baseline. Its checkpoint remained 4.116 MB, and measured latency was 0.768 ms. The implementation zeroed filters but retained the original layer dimensions, so this run did not show a reduction in dense checkpoint size or parameter count.

 Smaller student and distillation

The StudentCNN has two convolutional layers (16 and 32 output channels) and a 64-unit hidden layer. Its results compared with the baseline are:

| Model | Parameters | Test accuracy | Checkpoint size | CPU latency |
|---|---:|---:|---:|---:|
| Baseline CNN | 1,077,898 | 91.45% | 4.116 MB | 0.800 ms |
| StudentCNN, trained normally | 105,866 | 90.99% | 0.406 MB | 0.335 ms |
| StudentCNN, distilled | 105,866 | 90.00% | 0.406 MB | 0.335 ms |

The normally trained student used about 9.8% of the baseline’s parameters and checkpoint size—approximately a 90.2% reduction in each—and had about 2.4 times lower measured latency. Its test accuracy was 0.46 percentage points below the baseline.

The distilled student used temperature 4 and an equal weighting of the label loss and teacher-matching loss. It reached 90.00% accuracy, 0.99 percentage points below the normally trained student and 1.45 points below the baseline. Because both students have the same architecture, their parameter count, file size, and measured latency were the same. In these results, distillation did not improve accuracy over training the student directly.

 Conclusion

The measurements identify the compact StudentCNN as the clearest size-and-latency reduction: it substantially reduced parameters, checkpoint size, and CPU inference time while retaining accuracy near the baseline. Unstructured pruning produced the highest observed accuracy, 92.32% at both 50% and 75% sparsity, but did not improve measured dense inference latency. At 90% sparsity, the sparse-format checkpoint was smaller than the dense one, but remained slower than the compact student in the reported latency measurements. Structured pruning’s saved dense model remained approximately baseline-sized because the architecture was not rebuilt with fewer channels. Knowledge distillation was tested, but the distilled student scored lower than the student trained with labels alone. Overall, the results show that accuracy, sparsity, checkpoint size, parameter count, and runtime are separate metrics; each must be measured to determine whether a compression method is effective.
