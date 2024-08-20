Adverse Weather Creation using Unpaired Image-
to-Image Translation*

 
Sankalp Joshi 


 
 
 
 
 
Abstract— This project aims to develop an unsupervised image-to-image translation model that can effectively translate images from adverse weather conditions to standard conditions, enhancing the performance of autonomous systems in various environments. The proposed solution employs generative adversarial networks (GANs) to generate realistic low-visibility nighttime scenarios from high-visibility daytime datasets. The model is designed to disentangle invariant and variant features without relying on supervision or task-specific knowledge.The key challenges addressed in this work include the lack of availability of precisely aligned paired datasets, maintaining semantic consistency during translation, and balancing the trade-off between generating diverse synthetic images and preserving the original content and structure. The project will leverage the Berkeley DeepDrive (BDD100K) dataset for autonomous vehicles and the Landing Approach Runway Detection (LARD) dataset for aviation applications.
The central concept revolves around harnessing generative models to proficiently generate realistic low-visibility nighttime scenarios for autonomous systems by learning from available high-visibility daytime datasets. The performance and viability of the methodologies will be systematically evaluated against classical approaches, and user feedback will be incorporated to improve the model's performance and usability.The proposed solution aims to address the challenges in unsupervised or weakly-supervised learning settings, where the acquisition of precisely aligned pairs of images captured at different times and weather conditions is impractical. By accurately disentangling invariant and variant features, the model can effectively translate images between adverse and standard conditions, enhancing the robustness and reliability of autonomous systems operating in diverse environments.
Keywords— GAN, Unsupervised ,Image-to-Image Translation, Semantic Image Segmentation,Autonomous Systems,Adverse Weather Conditions
INTRODUCTION 
The objective of this project is to generate synthetic images representing critical low- visibility scenarios for autonomous vehicles and aviation systems. These images will serve as a repository to enhance perception algorithms and enable navigation in dynamic real- world environments. The focus will be on generating low-visibility nighttime scenario images by performing style transfers on high-visibility daytime images or employing alternative approaches .The primary objective of this project is to investigate and implement advanced image-to-image translation techniques, specifically focusing on unpaired methods such as CycleGAN.The aim is to develop algorithms capable of transforming images from one domain to another without the need for paired examples in the training dataset. By achieving this objective, the project aims to address the limitations posed by conventional paired methods, particularly in scenarios involving extreme weather conditions, low visibility, and other complex environments. Ultimately, the goal is to advance the capabilities of computer vision systems for applications in autonomous vehicles, aviation systems, surveillance, and industrial inspections.
MOTIVATION
The motivation behind this project stems from the critical need for robust image translation techniques that can operate effectively in real-world scenarios where obtaining paired training data is impractical or prohibitively expensive. By leveraging unpaired image-to-image translation methods such as CycleGAN and QGANs, the project seeks to overcome these challenges and enable the generation of realistic images across diverse domains. The potential applications of such techniques in fields like autonomous driving, surveillance, and industrial inspection further drive the motivation for this research. Additionally, the project aims to explore the intersection of classical deep learning techniques and quantum computing, presenting an exciting opportunity to push the boundaries of image synthesis capabilities 
Image-to-image translation is a fundamental task in computer vision, with applications ranging from style transfer to semantic segmentation. Traditional methods rely on paired datasets, where each input image is associated with a corresponding output image, for training. However, acquiring such paired data can be difficult, if not impossible, in many real-world scenarios. Unpaired image-to-image translation techniques have emerged as a solution to this challenge, enabling the transformation of images between different domains without the need for paired examples. CycleGAN, introduced by Zhuet al. in 2017, is a prominent example of such a technique, utilizing cycle-consistent adversarialnetworks to learn mappings between domains, leveraging the principles of quantum computing to produce diverse and realistic images. By building upon the foundations laid by CycleGAN and exploring the potential of GANs, this project aims to contribute to the advancement of image synthesistechniques for practical applications in various industries.
Eg. One collection may comprise images of clear daytime driving, while another contains images of snowy or nighttime driving scenes. By employing the CycleGAN architecture, with each GAN consisting of a discriminator and a generator model, we can effectively translate images from one driving condition to another. GAN-1 can focus on translating clear driving conditions to adverse ones, while GAN-2 handles the reverse translation. This project aims to replicate the successful outcomes achieved through CycleGAN for such driving conditions.
Classical Generative Modelling Methods – CycleGAN The CycleGAN (Cycle-Consistent Generative Adversarial Network) framework has been employed for unpaired image-to-image translation. This approach utilizes two generative adversarial networks (GANs): one for forward translation, converting images from the source domain to the target domain, and the other for inverse translation. Additionally, cycle-consistency loss is introduced to ensure that reconstructed images maintain essential features of the originals, thereby enforcing consistency between translated and reconstructed images. During training, generators aim toproduce realistic images in the target domain while deceiving discriminators, which distinguish between real and translated images. The cycle-consistency loss encourages generators to learn meaningful mappings between domains, even without paired examples.
Proposed Scalability: 
The proposed GAN solution offers several advantages and potential applications in the industrial context –
i.	Robust Night-time Image Generation – GANs can generate high-quality night-time images even in challenging low-light conditions, which is valuable for applications such as autonomous driving, surveillance, and industrial inspection
ii.	Data Augmentation and Synthetic Data Generation – The generated night-time images can be used for data augmentation and synthetic data generation, enabling more robust training of computer vision models in various industries.
iii.	Scalability and Parallelism – Quantum computing offers inherent parallelism and scalability advantages. As quantum hardware continues to evolve, GANs can leverage these advancements to generate night-time images more efficiently and at larger scales.
II.	MODEL DESCRIPTION
The Generative Adversarial Network (GAN) has gained widespread popularity due to its versatility and remarkable outcomes across various applications, such as text-to-image and image-to-image translation. This report explores CycleGAN, a specific type of GAN designed for unpaired image-to-image translation. CycleGAN's architecture involves training two generator models and two discriminator models simultaneously. The discriminator (D) distinguishes between real and fake images, while the generator (G) learns the data distribution, setting the two neural networks in opposition.

Unpaired image-to-image translation is a crucial technique in computer vision and machine learning, enabling the transformation of images from one domain to another without needing paired examples in the training dataset. Traditional image-to-image translation tasks, like style transfer or image colorization, require paired datasets, which are often difficult to obtain in real-world scenarios, such as extreme weather conditions, complex scenes, low visibility, and night-time settings. Safety concerns and logistical constraints frequently make acquiring paired datasets impractical. Unpaired image-to-image translation techniques address this challenge by learning mappings between images from different domains, such as transitions between clear and foggy weather, daylight and night-time scenes, or varying visibility conditions.
III.	RELATED WORKS
Various approaches have been developed to tackle computer vision challenges such as unpaired image-to-image translation, low-light image enhancement, adverse weather vision tasks, and uncertainty-aware learning.

Unpaired image-to-image translation techniques, like CycleGAN [7], convert images between domains without paired training data but lack strong disentanglement abilities. UNIT [8] introduces shared latent spaces for better disentanglement, while MUNIT and DRIT further decompose images into domain-invariant content and domain-specific styles [3]. StarGAN enhances diversity through multi-domain translation but requires style code specification [5].For low-light image enhancement, EnlightenGAN improves luminosity without paired data but may not emphasize crucial foreground objects.

A common gap in unpaired image-to-image translation is the limited disentanglement of semantic information during translation. Models like CycleGAN [7] and UNIT [8] often fail to enforce semantic preservation, crucial for applications like autonomous driving in adverse weather [6]. Although MUNIT and DRIT address this by decomposing visual information [3], further advancements are needed for robust image translations [5].

Our focus is on translating entire images to daylight while enhancing weak object signals in dark conditions.Adverse weather vision tasks involve challenges like localization and semantic segmentation under varying conditions. ToDayGAN and Porav et al.'s methods improve image quality for these tasks by transforming images to ideal conditions [12].In unsupervised image-to-image translation, StarGAN and AttGAN enable multi-domain translation with target attribute vectors, while UNIT emphasizes shared latent spaces [1][4]. MUNIT and DRIT enhance diversity through disentangled representations [3].Adverse weather enhancement is crucial for tasks like semantic segmentation and object detection. Models like EnlightenGAN and ForkGAN improve adverse weather images using image-to-image translation techniques [13].Uncertainty-aware learning uses heteroscedastic regression to model observation uncertainty, useful in high-noise regions like rainy night images, improving robustness in visual tasks [14][15].
Another gap lies in adverse weather image enhancement for semantic segmentation and object detection. While ToDayGAN [12] and Porav et al.'s method [11] enhance image quality for specific tasks, they often overlook holistic improvement necessary for effective scene understanding. Existing techniques focus on luminosity or ideal conditions, neglecting visual distortions from rain and fog. Comprehensive, domain-specific enhancement techniques are needed to address these challenges in autonomous driving scenarios [10][13].
I.	Overview of the Work
I.	Objectives of the Paper
The primary objectives of this paper are to:
Develop an Unsupervised Image-to-Image Translation Model: The paper aims to create a model that can effectively translate images from adverse weather conditions to standard conditions, enhancing the performance of autonomous systems in various environments.
Improve Performance in Scene Understanding Tasks: The model is designed to improve the performance of scene understanding tasks such as semantic segmentation and object detection under adverse weather conditions.
Enhance Image Quality and Structure: The paper focuses on enhancing the image quality and structure necessary for effective scene understanding in adverse weather conditions, which is often overlooked in existing techniques.


 
Disentangle Invariant and Variant Features: The model aims to accurately disentangle invariant and variant features without relying on supervision or task-specific knowledge.

Key Challenges and Gaps
The paper identifies the following key challenges and gaps in the current research landscape:
Limited Availability of Precisely Aligned Paired Datasets: The acquisition of precisely aligned pairs of images captured at different times and weather conditions is impractical, making it challenging to develop effective image-to-image translation models.
Neglect of Complex Visual Distortions: Existing techniques predominantly focus on enhancing luminosity or transforming images to ideal conditions, neglecting the complex visual distortions caused by adverse weather phenomena such as rain and fog.
Limited Robustness and Accuracy: Current methods often overlook the holistic improvement of image clarity and structure necessary for effective scene understanding in adverse weather conditions, leading to limited robustness and accuracy
Proposed Solution
The proposed solution involves:
Employing Generative Modeling: The paper utilizes generative models, specifically Generative Adversarial Networks (GANs), to enhance efficiency in representing and sampling complex distributions.
Evaluating Model Efficacy: The model's efficacy is evaluated using metrics such as Fréchet Inception Distance (FID) and computational complexity, resource requisites, and scalability.
Datasets:
For autonomous vehicles: Leveraging the Berkeley DeepDrive dataset (BDD100K), comprising approximately 70,000 training images.
For aviation: Utilizing the Landing Approach Runway Detection (LARD) dataset, encompassing roughly 15,000 real and synthetic runway images.
Proposed Solution:

Design Approach and Details
This section presents a detailed description of the CycleGAN framework utilized for image-to-image translation without paired training data. The model architecture and training methodology are discussed, emphasizing the generators, discriminators, and loss functions employed. Implementation was carried out using PyTorch, with RGB images of size 256x256 represented as tensors.

Cycle Generative Adversarial Network (CycleGAN)
CycleGAN represents a significant advancement in unsupervised image translation tasks, introduced to overcome the limitations of paired training datasets. Unlike traditional GANs that require aligned pairs of images for supervised learning, CycleGAN operates by learning mappings between two domains, X and Y, using adversarial and cycle consistency losses. The generators (G: X → Y and F: Y → X) are tasked with translating images between domains, while the discriminators (DY and DX) distinguish between real and generated images, ensuring the fidelity of translated outputs. Adversarial and Cycle Consistency Losses
The core of CycleGAN's training lies in its loss functions:
Adversarial Loss: Ensures that generated images are indistinguishable from real images in the target domain, formulated to minimize the discriminators' ability to differentiate between real and generated samples.
Cycle Consistency Loss: Mitigates mode collapse and enhances image quality by enforcing that an image translated to the target domain and back retains its original characteristics. This loss function aids in maintaining semantic consistency across domains.
Architectural Components
CycleGAN's architecture includes:
Generator: Composed of encoding (downsampling), transformation (residual blocks), and decoding (upsampling) stages, designed to effectively translate images across domains while preserving image quality.
Discriminator: Utilizes convolutional layers to assess the authenticity of generated images, ensuring that they align with the distribution of real images in their respective domains.


 

Fig1.Working of a GAN
Results and Discussion

We utilized CycleGAN to transform images, converting daytime scenes to nighttime and clear driving conditions to rainy weather. After 25 epochs of training on a dataset of 114 clear driving condition images and 78 adverse weather images, we achieved promising results. However, training was limited to 50 epochs due to time constraints. Our model demonstrated success in mapping images from clear to adverse conditions, albeit with sensitivity to hyperparameters.We take the Berkeley dataset and utilize 506 images and take 35 epochs



Results after 10 epochs and 35 epoch respectively

 
Rain as adverse Condition

For image synthesis in autonomous vehicles and aviation systems, we explored Generative Adversarial Networks (GANs). These networks leverage circuits to encode image features, enabling the generation of diverse, realistic scenarios. Our feasibility study evaluates GANs' ability to produce synthetic images reflecting adverse weather conditions, nighttime visibility challenges, and complex obstacle configurations.

Conclusion and Future Directions:
The paper presents a comprehensive overview of the challenges and gaps in unsupervised image-to-image translation, particularly in the context of adverse weather conditions. The proposed solution employs generative models, specifically CycleGAN, to translate images from adverse weather conditions to standard conditions. The model is designed to disentangle invariant and variant features without relying on supervision or task-specific knowledge. The performance and viability of the methodology are evaluated against classical approaches, and user feedback is incorporated to improve the model's performance and usability.

Further advancements in disentanglement techniques are necessary to achieve more robust and accurate image translations.It emphasizes the need for comprehensive and domain-specific image enhancement techniques tailored to address the unique challenges posed by adverse weather conditions in autonomous driving scenarios,it also suggests exploring alternative methods such as paired image-to-image translation techniques, unsupervised image-to-image translation methods, and adversarial learning techniques to address the challenges in image-to-image translation.The paper highlights the importance of uncertainty-aware learning in addressing the challenges of image-to-image translation under adverse weather conditions.The paper also emphasizes the need for scaling up the proposed solution to handle larger datasets and integrating it with other techniques to enhance its performance and usability.
REFERENCES
[1]	H. Emami, M. M. Aliabadi, M. Dong and R. B. Chinnam, "SPA-GAN: Spatial Attention GAN for Image-to-Image Translation," in IEEE Transactions on Multimedia, vol. 23, pp. 391-401, 2021, doi: 10.1109/TMM.2020.2975961.

[2]	K. Bian, S. Zhang, F. Meng, W. Zhang, and O. Dahlsten, "Symmetry-guided gradient descent for quantum neural networks," arXiv:2404.06108 [quant-ph] (April 2024).

[3]	Lakhanpal, S., Jaipuria, A., Banerjee, S., & Pandey, S. (2020). A Review on Image to Image Translation using Generative Adversarial Networks. International Research Journal of Engineering and Technology(IRJET), 07(12), 104-105.

[4]	Ko, K., Yeom, T., & Lee, M. (2023). SuperstarGAN: Generative adversarial networks for image- to-image translation in large-scale domains. Neural Networks, 162, 330-339. DOI: 10.1016/j.neunet.2023.02.042.

[5]	Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Pro- ceedings of the 27th International Conference on Neural Information Processing Systems
- Volume 2, NIPS’14, pages 2672–2680, Cambridge, MA, USA, 2014. MIT Press.

[6]	P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-Image Translation with Conditional Adversarial Networks," in 2017 IEEE Conference on Computer Vision and Pattern Recognition	(CVPR),	2016,	pp.	5967-5976.	Available: https://api.semanticscholar.org/CorpusID:6200260

[7]	J. -Y. Zhu, T. Park, P. Isola and A. A. Efros, "Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 2242-2251, doi: 10.1109/ICCV.2017.244.

[8]	Liu, M.Y., Breuel, T., Kautz, J.: Unsupervised image-to-image translation networks. In: Advances in Neural Information Processing Systems. pp. 700–708 (2017)

[9]	D. G. Lowe, "Object recognition from local scale-invariant features," Proceedings of the Seventh IEEE International Conference on Computer Vision, Kerkyra, Greece, 1999, pp. 1150-1157 vol.2, doi: 10.1109/ICCV.1999.790410.

[10]	Milford, M.J., Wyeth, G.F.: Seqslam: Visual route-based navigation for sunny summer days and stormy winter nights. In: 2012 IEEE International Conference on Robotics and Automation. pp. 1643–1649. IEEE (2012)

[11]	Porav, H., Bruls, T., Newman, P.: Don’t worry about the weather: Unsupervised condition-dependent domain adaptation. In: 2019 IEEE

[12]	Intelligent Transportation Systems Conference (ITSC). pp. 33–40. IEEE (2019)

[13]	Porav, H., Maddern, W., Newman, P.: Adversarial training for adverse conditions: Robust metric localisation using appearance transfer. In: 2018 IEEE International Conference on Robotics and Automation (ICRA). pp. 1011–1018. IEEE (2018)

[14]	Romera, E., Bergasa, L.M., Yang, K., Alvarez, J.M., Barea, R.: Bridging the day and night domain gap for semantic segmentation. In: 2019 IEEE Intelligent Vehicles Symposium (IV). pp. 1312–1318. IEEE (2019)

[15]	Ros, G., Alvarez, J.M.: Unsupervised image transformation for outdoor semantic labelling. In: 2015 IEEE Intelligent Vehicles Symposium (IV). pp. 537–542. IEEE (2015)

[16]	Sakaridis, C., Dai, D., Van Gool, L.: Semantic foggy scene understanding with synthetic data. International Journal of Computer Vision 126(9), 973–992 (2018)
 

