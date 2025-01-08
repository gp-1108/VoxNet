<div align="center" style="position: relative;">
<img src="https://img.icons8.com/?size=512&id=55494&format=png" align="center" width="30%" style="margin: -20px 0 0 20px;">
<h1>VOXNET</h1>
<p align="center">
	<em>VoxNet: Where 3D Data Speaks Volumes!</em>
</p>
</div>
<br clear="right">

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#🤖-usage)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

VoxNet is an innovative project designed to efficiently preprocess and analyze 3D volumetric data, particularly from the ModelNet40 dataset. Key features include automated data handling, a robust 3D convolutional neural network architecture, and comprehensive logging for monitoring. Ideal for researchers and developers in computer vision and deep learning applications, it streamlines the workflow for advanced 3D model classification.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes a 3D convolutional neural network (ResBNVox64Net) for efficient processing of volumetric data.</li><li>Incorporates convolutional layers, residual blocks, and pooling operations to optimize feature extraction.</li><li>Visual representations of the architecture enhance clarity for developers and stakeholders.</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Ensures logging functionality for effective monitoring and troubleshooting.</li><li>Utilizes consistent coding standards across multiple languages (Python, Shell, etc.).</li><li>Maintains a robust logging mechanism for operational visibility and reliability.</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive documentation of code functionalities and architecture.</li><li>Includes usage commands and installation instructions for ease of use.</li> |
| 🔌 | **Integrations**  | <ul><li>Integrates with dataset management tools for preprocessing and conversion (e.g., ModelNet40 dataset).</li><li>Supports multiple scripts to facilitate diverse tasks within the project.</li><li>Utilizes job scheduling (e.g., SLURM) for efficient resource management.</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Project components are organized into separate files for clarity and ease of maintenance.</li><li>Each module serves distinct functional purposes, enhancing reusability.</li><li>Encourages a structured approach to code organization, facilitating scalability.</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimized for handling 3D input data effectively.</li><li>Efficient learning and generalization capabilities across various tasks.</li><li>Employs best practices in coding and architecture to enhance system performance.</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Key dependencies include Python, SLURM, PyTorch and specific model architecture definitions.</li><li>Utilizes environment configuration files for consistent setup.</li><li>Minimizes external dependencies to enhance portability.</li></ul> |

---

## 📁 Project Structure

```sh
└── VoxNet/
	├── requirements.txt
    ├── ModelNet40Voxel.zip
    ├── ModelNet40Voxel64.zip
    ├── dataset_converter.py
    ├── dataset_creation
    │   ├── dataset_converter.py
    │   ├── env.def
    │   ├── job.slurm
    │   └── script.sh
    ├── img_nn_architecture.py
    ├── logger.py
    ├── plots_extraction
    │   └── plot_extraction.py
    ├── resbnvox64net_architecture
    ├── resbnvox64net_architecture.png
    ├── script.sh
    └── training
        ├── CustomFocalLoss.py
        ├── Dataset.py
        ├── Trainer.py
        ├── env.def
        ├── job.slurm
        ├── job_32.slurm
        ├── models
        │   ├── BaseVoxNet.py
        │   ├── BatchNormVoxNet.py
        │   ├── ResBNVox64Net.py
        │   ├── ResBNVoxNet.py
        │   ├── ResVox64Net.py
        │   ├── ResVoxNet.py
        │   └── __init__.py
        ├── rotation_test_script.py
        ├── script.sh
        ├── script_32.sh
        └── train.py
```


### 📂 Project Index
<details open>
	<summary><b><code>VOXNET/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/script.sh'>script.sh</a></b></td>
				<td>- Facilitates the preprocessing and conversion of the ModelNet40 dataset for use in the VoxNet project<br>- It automates the cleanup, downloading, unzipping, and processing of the dataset, ensuring the data is prepared efficiently for subsequent tasks<br>- Additionally, it compresses the converted dataset for easy storage and access within the overall codebase architecture.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/img_nn_architecture.py'>img_nn_architecture.py</a></b></td>
				<td>- Generate a visual representation of the ResBNVox64Net architecture, detailing the flow and structure of the model's components, including convolutional blocks, residual blocks, pooling, and fully connected layers<br>- This diagram serves as a valuable reference for understanding the overall design and decision-making process within the codebase, enhancing clarity for developers and stakeholders engaged in the project.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/resbnvox64net_architecture'>resbnvox64net_architecture</a></b></td>
				<td>- Defines the architecture for a 3D convolutional neural network designed for processing volumetric data<br>- By utilizing a series of convolutional layers, residual blocks, and pooling operations, the network effectively extracts features before classifying them through fully connected layers<br>- This structure enables efficient learning and generalization across various tasks in the project, emphasizing performance in handling 3D input.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/logger.py'>logger.py</a></b></td>
				<td>- Logging functionality is implemented to facilitate monitoring and troubleshooting within the application<br>- It establishes a robust logging mechanism that captures various log levels, ensuring logs are both saved to a file and displayed in the console<br>- By managing log rotation and formatting, it enhances the codebase's maintainability and operational visibility, contributing to overall system reliability and performance in the VoxNet project.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/dataset_converter.py'>dataset_converter.py</a></b></td>
				<td>- The `dataset_converter.py` file serves a crucial role within the project architecture by facilitating the conversion and management of datasets used in the application<br>- Its main purpose is to preprocess and prepare data for further analysis or modeling, ensuring that datasets are in the appropriate format and optimized for performance<br>- This file achieves its objectives through various functionalities such as logging processing activities, handling multiprocessing for efficiency, and utilizing libraries for data manipulation and compression<br>- By centralizing these tasks, `dataset_converter.py` helps maintain the overall integrity and usability of the codebase, allowing other components of the project to seamlessly access and utilize the processed datasets<br>- In essence, it acts as a foundational tool for data handling, which is vital for the success of the project's workflows.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- training Submodule -->
		<summary><b>training</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/script.sh'>script.sh</a></b></td>
				<td>- Facilitates the preparation and execution of a training process for a deep learning model within the VoxNet project<br>- It manages the cleanup, downloading, and unzipping of the ModelNet40Voxel dataset, then initiates the model training using a specified script and environment, ensuring a streamlined workflow for users to efficiently train their models with the necessary data and resources.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/CustomFocalLoss.py'>CustomFocalLoss.py</a></b></td>
				<td>- CustomFocalLoss implements a specialized loss function designed to address class imbalance in training neural networks<br>- By adjusting the contribution of each class to the overall loss through a focal mechanism, it enhances model focus on harder-to-classify examples<br>- This component plays a crucial role in the project's architecture, ensuring improved learning outcomes in scenarios with skewed class distributions.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/Trainer.py'>Trainer.py</a></b></td>
				<td>- Facilitates the training and evaluation of machine learning models through a structured Trainer class<br>- By implementing k-fold cross-validation and early stopping, it optimizes model performance while preventing overfitting<br>- It manages the training loop, validates model accuracy, and saves the best-performing model, ultimately ensuring efficient and effective training within the broader codebase architecture.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/train.py'>train.py</a></b></td>
				<td>- Facilitates the training of various VoxNet models on the ModelNet40 dataset by systematically configuring and executing experiments<br>- It dynamically generates model names based on parameters, manages dataset loading and augmentation, and initializes training components such as optimizers and loss functions<br>- Ultimately, it produces trained model outputs for analysis and further development within the broader project architecture focused on 3D object recognition.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/env.def'>env.def</a></b></td>
				<td>- Sets up a lightweight environment based on Ubuntu 20.04, ensuring a streamlined installation of essential Python packages and tools<br>- This configuration enables efficient development and execution of machine learning models and data processing tasks within the broader project architecture, facilitating seamless integration and functionality across various components of the codebase.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/rotation_test_script.py'>rotation_test_script.py</a></b></td>
				<td>- Facilitates the evaluation of a trained model on the ModelNet40 dataset by applying rotation transformations to voxel grids<br>- It calculates prediction accuracy through a confusion matrix, showcasing the model's performance across various classes<br>- Additionally, it generates a visual representation of the confusion matrix, aiding in the analysis of classification results within the broader architecture of the project.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/script_32.sh'>script_32.sh</a></b></td>
				<td>- Facilitates the setup and execution of a training environment for the VoxNet project by managing the dataset and model outputs<br>- It automates the cleanup of previous datasets, downloads necessary files from Google Drive, unzips the dataset, and initiates the training process using a Singularity container, ensuring a streamlined workflow for model training and evaluation within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/job_32.slurm'>job_32.slurm</a></b></td>
				<td>- Facilitates job submission for a deep learning training process within a high-performance computing environment<br>- Configured to manage resources effectively, including memory and GPU allocation, while ensuring output and error logging<br>- Integrates seamlessly into the project structure by initiating the necessary training script, supporting the overall architecture aimed at efficient model development and experimentation in voxel-based neural networks.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/job.slurm'>job.slurm</a></b></td>
				<td>- Facilitates job scheduling for training a voxel-based neural network model by configuring SLURM parameters, such as job name, error and output file management, resource allocation, and email notifications<br>- It ensures the execution of a training script within a specified environment, optimizing computational resources for effective model training within the overall project architecture focused on deep learning applications.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/Dataset.py'>Dataset.py</a></b></td>
				<td>- Facilitates the management and processing of 3D voxel grid datasets for machine learning applications, specifically targeting the ModelNet40 dataset<br>- It organizes data into training and testing splits, supports transformations, and allows for data augmentation through rotations<br>- The code also includes functionality for decompressing and deserializing data, ensuring efficient retrieval and preparation of samples for model training.</td>
			</tr>
			</table>
			<details>
				<summary><b>models</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/ResVox64Net.py'>ResVox64Net.py</a></b></td>
						<td>- Defines a 3D convolutional neural network architecture, ResVox64Net, which is tailored for multi-class classification tasks<br>- Comprising initial convolutional layers followed by a series of residual blocks, the model captures complex features from volumetric data<br>- It culminates in fully connected layers to output class predictions, thereby facilitating effective analysis and interpretation of 3D input data within the overall project framework.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/BatchNormVoxNet.py'>BatchNormVoxNet.py</a></b></td>
						<td>- BatchNormVoxNet serves as a neural network architecture designed for 3D data classification<br>- By integrating convolutional layers with batch normalization and dropout techniques, it enhances model performance and stability<br>- The architecture effectively extracts features from volumetric data, ultimately producing class predictions, making it essential for applications in fields such as medical imaging and 3D object recognition within the overall project framework.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/BaseVoxNet.py'>BaseVoxNet.py</a></b></td>
						<td>- BaseVoxNet serves as a foundational neural network model designed for 3D data classification tasks within the broader project framework<br>- It incorporates multiple convolutional layers, pooling operations, and fully connected layers, effectively transforming input data into class predictions<br>- By employing dropout and activation functions, it enhances learning robustness, making it a pivotal component for training effective machine learning models in the codebase.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/ResBNVoxNet.py'>ResBNVoxNet.py</a></b></td>
						<td>- Defines a 3D convolutional neural network architecture, ResBNVoxNet, designed for classifying volumetric data into specified categories<br>- Integrating multiple residual blocks enhances feature extraction through skip connections, while batch normalization and dropout layers improve training stability and prevent overfitting<br>- This model serves as a crucial component within the broader project, aimed at tackling complex deep learning tasks involving 3D data.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/ResVoxNet.py'>ResVoxNet.py</a></b></td>
						<td>- Defines the ResVoxNet architecture, a 3D convolutional neural network designed for classification tasks<br>- It incorporates residual blocks to enhance feature extraction and mitigate vanishing gradients, processing volumetric input data effectively<br>- The model employs convolutional, pooling, and fully connected layers, culminating in a classification output tailored for multiple classes, thereby facilitating advanced analyses in fields such as medical imaging or volumetric data interpretation.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/training/models/ResBNVox64Net.py'>ResBNVox64Net.py</a></b></td>
						<td>- ResBNVox64Net serves as a 3D convolutional neural network model designed for classification tasks within the broader project architecture<br>- By leveraging residual blocks and pooling layers, it efficiently processes volumetric data, enhancing feature extraction while maintaining spatial hierarchies<br>- Ultimately, this model outputs class predictions, contributing significantly to the project's goal of robust data analysis and interpretation.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- dataset_creation Submodule -->
		<summary><b>dataset_creation</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/dataset_creation/script.sh'>script.sh</a></b></td>
				<td>- Facilitates the creation and preparation of the ModelNet40 dataset for use within the VoxNet project<br>- It automates the download, extraction, and conversion of 3D model data into a voxel representation suitable for machine learning tasks<br>- Additionally, it organizes the processed data into a compressed format ready for integration into the main codebase, streamlining the workflow for data handling and model training.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/dataset_creation/env.def'>env.def</a></b></td>
				<td>- Facilitates the setup of a development environment by defining a Docker configuration that utilizes Ubuntu 20.04<br>- It automates the installation of essential utilities, Python, and key libraries such as NumPy, SciPy, Trimesh, and Zstandard<br>- This environment is crucial for ensuring that subsequent data manipulation and analysis tasks within the project run seamlessly and efficiently.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/dataset_creation/job.slurm'>job.slurm</a></b></td>
				<td>- Facilitates job submission for a voxel generation process within the dataset creation module of the project<br>- By leveraging SLURM job scheduling, it manages resource allocation and execution parameters, ensuring efficient processing under specified constraints<br>- The script navigates to the appropriate directory and triggers the primary generation script, thereby streamlining the workflow and enhancing the overall functionality of the codebase architecture.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/dataset_creation/dataset_converter.py'>dataset_converter.py</a></b></td>
				<td>- The `dataset_converter.py` file is an integral component of the project's architecture, primarily focused on the creation and conversion of datasets<br>- Its main purpose is to facilitate the transformation of raw data into a structured format suitable for analysis or machine learning applications<br>- By leveraging libraries such as NumPy and SciPy, it optimizes data processing tasks, including resizing and filling holes in datasets<br>- Moreover, the file incorporates a logging mechanism designed for multiprocessing environments, ensuring that all operations are efficiently recorded for debugging and monitoring purposes<br>- This enhances the overall reliability and maintainability of the codebase<br>- In summary, `dataset_converter.py` serves as a critical utility for dataset preparation, contributing to the project's goal of enabling seamless data management and analysis workflows.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- plots_extraction Submodule -->
		<summary><b>plots_extraction</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/gp-1108/VoxNet/blob/master/plots_extraction/plot_extraction.py'>plot_extraction.py</a></b></td>
				<td>- Facilitates the analysis and visualization of training logs by parsing performance metrics for machine learning models across multiple folds<br>- It generates and saves detailed learning curves and accuracy progression plots, alongside combined visualizations, in designated directories<br>- Additionally, it exports the parsed results as JSON files, streamlining the evaluation process within the overall project architecture.</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with VoxNet, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Libraries:** The project requires the following libraries to be installed:
```sh
graphviz==0.20.3
matplotlib==3.10.0
numpy==2.2.1
scikit_learn==1.6.0
scipy==1.15.0
torch==2.5.1+cpu
tqdm==4.67.1
trimesh==4.5.3
zstandard==0.23.0
```


### ⚙️ Installation

Install VoxNet using one of the following methods:

**Build from source:**

1. Clone the VoxNet repository:
```sh
❯ git clone https://github.com/gp-1108/VoxNet
```

2. Navigate to the project directory:
```sh
❯ cd VoxNet
```

3. Install the project dependencies:

```sh
❯ pip install -r requirements.txt
```



### 🤖 Usage
The various training and dataset creations scripts can be easily found in the repository. Feel free to use whatever you need.

## 🎗 License

This project is protected under the [AGPL 3.0](https://choosealicense.com/licenses/agpl-3.0/) License.

---

## 🙌 Acknowledgments

- The project was made possible by the overall effort of:
  - [Girotto Pietro](https://github.com/gp-1108)
  - [Enrico D'Alberton](https://github.com/enricopro)
  - [Yi Jian Qiu](https://github.com/Qiuzzo)
- The work is based on the original paper [Orientation-boosted Voxel Nets for 3D
Object Recognition](https://arxiv.org/abs/1604.03351)

---
