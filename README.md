# AMKSR-for-SAR-ATR

Note that the reviewer should first download the data sets under various scenarios to the same directory as the source code. Then, the reviewer can run the source code of AMKSR on different data sets to observe the recognition results of AMKSR.

The demo.m provides a case study about the entire training and test process of AMKSR. 

The training time under various scenarios is quite time-consuming.  For the convenience of the reviewer to observe the recogniiton results, we directly retain the relevant data (e.g., the features of data, the kernel matrices of data, and the weigths of different kernels) after the training and give the test process under various scenarios. Specifically, AMKSR_SOC.m, AMKSR_EOC_1.m, AMKSR_EOC_2.m, AMKSR_EOC_3.m, AMKSR_SAMPLE.m, and AMKSR_ACD.m are separatey used to verify the test results under SOC, EOC-1, EOC-2, EOC-3, SAMPLE, and SAR-ACD.
