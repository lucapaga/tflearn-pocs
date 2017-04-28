# MNIST Test

## Setup

1. MNIST TRAINING/TEST Set
2. DIGITS HANDWRITTEN BY ME (see [../MY_data] Folder)

## Commonz

- 784 INPUT Neurons (28x28 image)
- 128 Neurons on 1st HIDDEN LAYER
- 256 Neurons on 2nd HIDDEN LAYER
-  10 OUTPUT Neurons (10 digits, 10 classes)
- SOFTMAX on OUTPUT
- SOFTMAX CROSS ENTROPY
- LEARNING RATE: 0.001

## TEST 1

### Commonz

- 784 INPUT Neurons (28x28 image)
- 128 Neurons on 1st HIDDEN LAYER
- 256 Neurons on 2nd HIDDEN LAYER
-  10 OUTPUT Neurons (10 digits, 10 classes)
- SOFTMAX on OUTPUT
- SOFTMAX CROSS ENTROPY
- LEARNING RATE: 0.001

### Results

NR.  | EPOCHS | ACTIVATION FUNCTION | UNO      | UNOw     | DUE      | TRE      | QUATTRO  | CINQUE   | SEI      | SETTE    | OTTO     | NOVE
---: | :----: | ------------------- | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | --------:
 1   |   20   | RELU                | 8 ( 60%) | 0 ( 83%) | 2 ( 99%) | 3 (100%) | 4 ( 99%) | 3 ( 99%) | 6 ( 99%) | 2 ( 62%) | 3 (100%) | 2 ( 99%)
 2   |   20   | LINEAR              | 8 ( 88%) | 6 ( 39%) | 2 ( 99%) | 3 ( 99%) | 9 ( 54%) | 3 (100%) | 6 ( 99%) | 3 ( 94%) | 3 ( 99%) | 2 ( 86%)
 3   |   20   | SIGMOID             | 8 ( 94%) | 6 ( 54%) | 2 ( 99%) | 3 ( 99%) | 4 ( 99%) | 2 ( 92%) | 6 ( 99%) | 2 ( 99%) | 3 ( 99%) | 2 ( 73%)


 ## TEST 2

 ### Commonz

 -  784 INPUT Neurons (28x28 image)
 - 1568 Neurons on 1st HIDDEN LAYER
 - 1568 Neurons on 2nd HIDDEN LAYER
 -   10 OUTPUT Neurons (10 digits, 10 classes)
 - SOFTMAX on OUTPUT
 - SOFTMAX CROSS ENTROPY
 - LEARNING RATE: 0.001

 ### Results

 NR.  | EPOCHS | ACTIVATION FUNCTION | UNO      | UNOw     | DUE      | TRE      | QUATTRO  | CINQUE   | SEI      | SETTE    | OTTO     | NOVE
 ---: | :----: | ------------------- | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | --------:
  1   |   20   | RELU                | 8 ( 60%) | 0 ( 83%) | 2 ( 99%) | 3 (100%) | 4 ( 99%) | 3 ( 99%) | 6 ( 99%) | 2 ( 62%) | 3 (100%) | 2 ( 99%)
  2   |   20   | LINEAR              | 8 ( 88%) | 6 ( 39%) | 2 ( 99%) | 3 ( 99%) | 9 ( 54%) | 3 (100%) | 6 ( 99%) | 3 ( 94%) | 3 ( 99%) | 2 ( 86%)
  2   |   20   | SIGMOID             | 8 ( 94%) | 6 ( 54%) | 2 ( 99%) | 3 ( 99%) | 4 ( 99%) | 2 ( 92%) | 6 ( 99%) | 2 ( 99%) | 3 ( 99%) | 2 ( 73%)
