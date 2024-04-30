# Clinical Domain Knowledge-Derived Template Improves Post Hoc AI Explanations in Pneumothorax Classification
-   We propose a template-guided approach to incorporate the clinical knowledge of pneumothorax into model explanations generated by XAI methods, thereby enhancing the quality of these explanations.
> ### Please read our [article](https://arxiv.org/pdf/2403.18871) for further information.

## Overview
-   Figure 1a shows the overview of our template guidance as a plug-and-play module for existing XAI methods.
-   To depict the pleural space from the clinical experts’ view, a canonical lesion annotation by radiologists is extracted as the basis for template generation.
-   Then several morphological operations are implemented to further refine the pleural space - potential occurrence areas of pneumothorax.
-   After that, we shepherd the original explanations using the generated template region: Only the pixel within the template boundaries will be included in model focus areas.
-   Finally, focus areas with or without template guidance are compared with the ground truth lesion annotations.

## Template Generation
-   Figure 1b summarizes the details of template generation: Using one canonical lesion delineation as the starting point, the final template is generated by flipping, overlap, and dilation.
-   Selected by radiologists, the starting lesion delineation contours at least the pleural space on one side.
-   Then the step of flipping turns over the original lesion delineation horizontally to generate template covering the other side.
-   After that, considering the domain knowledge that pneumothorax potentially occurs in both the left and the right pleural space, the step of overlapping is implemented to spotlight both left and right pleural spaces.
-   To address the issue that the chest radiographs are captured at different distances and angles, dilation is introduced to eliminate the problem of deformation through enlarging the template area to cover a broader space and the final template is obtained.
![](https://github.com/Han-Yuan-Med/template-explanation/blob/main/Figure%201.png)
*Figure 1: Overview of the proposed template-guided explanation pipeline.*

## Code Usage
-   STEP(i): Run `Res-50 Training.py` and `VGG-19 Training.py` to develop pneumothorax classifier based on the architecture of Res-50 and VGG-19.
-   STEP(ii): Run `Saliency Map.py` to create XAI explanations based on Saliency Map with and without template guidance.
-   STEP(iii): Run `Grad-CAM.py` to create XAI explanations based on Grad-CAM with and without template guidance.
-   STEP(iv): Run `Integrated Gradients.py` to create XAI explanations based on Integrated Gradients with and without template guidance.

## Citation
* Yuan, H., Hong, C., Jiang, P., Zhao, G., Tran, N. T. A., Xu, X., ... & Liu, N. (2024). Clinical Domain Knowledge-Derived Template Improves Post Hoc AI Explanations in Pneumothorax Classification. arXiv preprint arXiv:2403.18871.
