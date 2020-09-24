# Hardware-Trojan-Detection
Machine Learning Techniques for Hardware Trojan Detection 

### The problem

- Rapid development of technology drives companies to design and fabricate their ICs in non-trustworthy outsourcing foundries to reduce the cost
- There is space for a synchronous form of virus, known as Hardware Trojan (HT), to be developed. HTs leak encrypted information, degrade device performance or lead to total destruction.

### Description of CasLab-HT algorithm

- We used the design tool, Design Compiler NXT from Synopsys for the dataset's feature extraction
- The features consist via area and power characteristics of the circuits. In total they were used 50 area and power features.
- 7 Machine Learning models for the detection and classification of Trojan Free and Trojan Infected circuits, based on Gate Level Netlist phase and features for Application Specific Integrated Circuit (ASIC) circuits.
