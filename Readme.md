# Data Mining Projects

- Author: LIN JIANING

- Date: 2018.5

- Course: Data Mining 

- Hardware: Laptop

- Language: MATLAB

- IDE: MATLAB

- Description: In the project, I finished 8 tasks and implement Histogram/CCV/PCA/DCT for image analysis. Project Code is in the `Code` folder. 

- Results(incomplete, complete result is in `Report.pdf` but the analysis of the result is in Chinese):

  - PCA

  |    name    |                      I1                      |                      I2                      |                      I3                      |
  | :--------: | :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
  |   origin   |            ![I1](./Report/I1.bmp)            |            ![I2](./Report/I2.bmp)            |            ![I3](./Report/I3.bmp)            |
  |   **1**    |                                              |                                              |                                              |
  |  reduced   |  ![pca_I1_0.8](./Report/pca/pca_I1_0.8.png)  |  ![pca_I2_0.8](./Report/pca/pca_I2_0.8.png)  |  ![pca_I3_0.8](./Report/pca/pca_I3_0.8.png)  |
  | threshold  |                     0.8                      |                     0.8                      |                     0.8                      |
  | dim(ratio) |                 6  (0.8003)                  |                 2  (0.8933)                  |                 17  (0.8007)                 |
  |   **2**    |                                              |                                              |                                              |
  |  reduced   |  ![pca_I1_0.9](./Report/pca/pca_I1_0.9.png)  |  ![pca_I2_0.9](./Report/pca/pca_I2_0.9.png)  |  ![pca_I3_0.9](./Report/pca/pca_I3_0.9.png)  |
  | threshold  |                     0.9                      |                     0.9                      |                     0.9                      |
  | dim(ratio) |                 14  (0.9046)                 |                 3  (0.9264)                  |                 44  (0.9010)                 |
  |   **3**    |                                              |                                              |                                              |
  |  reduced   | ![pca_I1_0.95](./Report/pca/pca_I1_0.95.png) | ![pca_I2_0.95](./Report/pca/pca_I2_0.95.png) | ![pca_I3_0.95](./Report/pca/pca_I3_0.95.png) |
  | threshold  |                     0.95                     |                     0.95                     |                     0.95                     |
  | dim(ratio) |                 26  (0.9522)                 |                 4  (0.9536)                  |                 80  (0.9509)                 |
  |   **4**    |                                              |                                              |                                              |
  |  reduced   |    ![pca_I1_1](./Report/pca/pca_I1_1.png)    |    ![pca_I2_1](./Report/pca/pca_I2_1.png)    |    ![pca_I3_1](./Report/pca/pca_I3_1.png)    |
  | threshold  |                      1                       |                      1                       |                      1                       |
  | dim(ratio) |                   264  (1)                   |                   264  (1)                   |                   264  (1)                   |

  - DCT

  | Quality Factor (Q) |                     I4                     |                     I5                     |                     I6                     |
  | :----------------: | :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
  |        0.1         | ![dct_I4_0.1](./Report/dct/dct_I4_0.1.png) | ![dct_I5_0.1](./Report/dct/dct_I5_0.1.png) | ![dct_I6_0.1](./Report/dct/dct_I6_0.1.png) |
  |         1          |   ![dct_I4_1](./Report/dct/dct_I4_1.png)   |   ![dct_I5_1](./Report/dct/dct_I5_1.png)   |   ![dct_I6_1](./Report/dct/dct_I6_1.png)   |
  |         10         |  ![dct_I4_10](./Report/dct/dct_I4_10.png)  |  ![dct_I5_10](./Report/dct/dct_I5_10.png)  |  ![dct_I6_10](./Report/dct/dct_I6_10.png)  |
  |        100         | ![dct_I4_10](./Report/dct/dct_I4_100.png)  | ![dct_I5_100](./Report/dct/dct_I5_100.png) | ![dct_I6_100](./Report/dct/dct_I6_100.png) |
