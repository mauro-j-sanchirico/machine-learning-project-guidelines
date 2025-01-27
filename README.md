# Guidelines for Academic Machine Learning Projects Aimed at Transition to Production Environments

This repository provides guidelines and best practices for academic machine learning (ML) projects aimed at transitioning to a production environment. The guidelines are intended to cover and apply to a broad span of ML methods and a similarly broad span of applications, with in-depth focus on deep learning enabled perception methods, i.e., methods for using deep neural networks for detecting, classifying, and keeping track of patterns in data. These guidelines are intended to help ML researchers and researchers applying ML to their fields design ML components in a manner which facilitates later developing and deploying robust systems which perform reliably on constantly changing real world data. These guidelines provide technical background, software best practices, machine learning best practices, and research guidance to this end. ML best practices covered include project planning, requirements formulation, ethics evaluation, recommendations for team building and budgeting, data pipeline practices, considerations for how sensors can affect data, model architecture selection techniques, hyper-parameter tuning techniques, recommendations for robustness, hardening, red-teaming, post-training scaling, and other evaluation steps. This guide is aimed at ML researchers interested in performing their research in a manner which will facilitate rapid transition to production environments, and practitioners in other fields researching applications of ML in their respective fields.

## 1. Introduction

### 1.1. Who Should Read this Guide

This guide is mainly targeted for advanced undergraduate students and beginning graduate students researching ML applications and interested in conducting their research in a manner which will facilitate later transition of the research to a production environment, i.e., an environment where the outcomes of ML models are visible to real world users or making an impact on real world users in some way. This target audience includes ML researchers and researchers in other fields applying ML to their fields. This guide assumes that readers are already familiar with ML basics and ML theory relevant to their projects and does not attempt to cover these topics. Readers are deferred to classical texts to cover ML basics and theory. While the guide is aimed at use by advanced undergraduate and beginning graduate students in guiding academic projects, it may be useful to any readers conducting ML research or design work and interested in following responsible practices in doing so. To avoid slowing down research by imposing all of the constraints and rigor of designing and deploying a production-grade ML system prematurely, this guide focuses on preparing ML research that is *ready for* expedited transition to production, rather than *immediate* deployment of production grade systems, which can have widely varying requirements across applications.

TODO: Cite classical ML texts

### 1.2. Pre-requisites

Readers of this guide should already be comfortable with the mathematical fundamentals behind modern ML, specifically those that underpin deep learning, including linear algebra, correlation, model fitting, optimization, and numerical methods. It is assumed readers are already proficient in Python and know how to set up local and remote Python development environments. Prior familiarity with tools including scikit-learn, torch, torch lightning, thunder, pandas, matplotlib, numpy, cupy, Open Neural Network Exchange (ONNX) is assumed but not strictly required to understand the guide. Examples of some of these tools are provided. Ability to quickly learn how to use tools such as Captum (neural network explainability), Lime (general explainability), and the Adversarial Robustness Toolkit (ART). Familiarity with software basics including version control, setting up an integrated development environment (IDE), coding, and programming language basics is fully assumed and these are not covered in this guide.

### 1.3. Motivation

Due to the relatively recent prevalence of artificial intelligence (AI) systems, we often hear of the significant impacts (both positive and negative) that the technology will have on our daily lives, and on broader societies. Familiar products, such as cell-phones and cars, are being augmented with AI capabilities, and AI is increasingly being used in critical industries, such as medicine and national defense. However, the tendency for prominent AI techniques to produce the correct answers for the wrong reasons still presents significant risks. For example, the use of AI in self-driving vehicles has brought many philosophical questions once confined to the classroom (such as the famous trolley problem in ethics) to the heart of engineering design discussions for the specialists who design these systems. Issues with all-weather sensor reliability, once a goal more important for military applications than civilian applications, now make headlines as self-driving cars must accurately discriminate people, objects, and other vehicles, despite challenging weather conditions such as rain, fog, and changes in lighting.

Similar to other technological advances that impacted critical applications, such as the prevalence of software in modern devices (even those we do not typically think of as computers), the invention of new modes of travel (e.g., sea and air travel), and even the use of electricity itself, the maturation of AI from a field of experimental study to a mature practice suitable for real-world deployment requires AI practitioners adopt and implement a repertoire of safeguards to protect the users impacted by these deployments. Practitioners need to adopt these safeguards both proactively, to prevent foreseen incidents involving AI technology, and reactively, to ensure lessons learned from previous implementations are not forgotten. The [AI Incident Database](https://incidentdatabase.ai/) maintains a growing list of incidents in which AI systems deployed to the real world have caused harms to individuals, groups, and societies. The incidents documented range from annoyances for users of systems, to harmful injustices, and tragic outcomes with injuries and lives lost.

This guide aims to help student researchers conduct ML research responsibly, with the impacts that future productionization of the technology they are developing will have on future users and broader societies in mind. Software best practices covered are intended to help students who may not be familiar with production coding practices write code in a way the maximizes maintainability and expedites transition to productions without taking the full time and effort required to implement a production grade system prematurely. Project planning, requirements formulation, team building, and budgeting guidance is provided to help students who might be used to being assigned a problem make the transition to selecting and formulating a problem to solve, and collaborating with others to solve it with an idea in mind of how much resources might be required to do so. Ethics evaluation guidance helps students evaluate if their use case is appropriate for ML techniques. Data pipeline practices and considerations for how sensors can affect data are provided to help students move from employing pre-labeled classical ML datasets to using messier data found in the wild. Model architecture selection techniques and hyper-parameter tuning are covered along with recommendations for robustness testing, hardening, and red-teaming are covered to help students move beyond checking model accuracy and ensure the models are robust to real world data diversity and drift.

### 1.4. Nomenclature

In this guide, we distinguish between *Requirements*, *Best Practices*, *Recommendations*, and *Techniques*.

* *Requirements* are hard requirements which all ML research aimed at production transition must follow.
* *Best Practices* are strong recommendations which practitioners should have a good reason for deviating from. Any deviation from a best practice must come with an explanation.
* *Recommendations* are suggestions that can be deviated from for equal alternatives.
* *Techniques* are steps in the ML process which require judgement in application. Techniques may be helpful or even critical for some problems but unhelpful in others. Data augmentation is an example of a technique which may be critical in some cases but not applicable to all problems and liable to cause worse results if applied inappropriately.

*Anti-patterns* (i.e., what NOT to do) are also reviewed at the end of this guide.

## 2. Software Tools and Practices

### 2.1. Recommended Toolchain

While beginning research projects might be confined to a Jupyter notebook or a small number of Python scripts, ML research projects ready for transition to production need to employ an ecosystem of tools at various stages of the development cycle. An example production workflow is outlined below. While there are many options, and many production ML projects use different tools (e.g., PyTorch and TensorFlow) the general idea is the same: decouple training and inference software when possible. Adopting some of this workflow early can help academic ML research transition to production faster.

<div align="center">
<img src="images/toolchain.png" alt="Alt Text" width="800">
<h4>Typical toolchain for production machine learning workflows.</h4>
</div>

#### Requirement: Decouple training and inference code.

For systems that do not need to learn online, it is helpful to decouple training and inference software. Training and inference software often must meet different requirements, and may even be developed by different teams with different skills on different systems in production settings. It also might be necessary to train a model once and then deploy to multiple different production environments. In the example above, we see a common workflow where training is performed in Python, then models and data are serialized for export to a production system where inference is performed in C++. As a compiled language, well-written C++ can make better use of system resources and ensure timing and memory requirements are met. Even if runtime and memory benefits are small, they are compounded when an application is hosted many times.

As an interpreted language, Python is faster to develop in and its host of packages facilitate machine learning development. Decoupling enables the developers or teams of developers maintaining both sides of the workflow (training and production) to optimize their components for their respective requirements. A decoupled workflow is one where changes to each column require minimal or ideally no changes to the others. Systems which implement online learning will have tighter coupling between training and inference (for example, both will be implemented in the production system) but should still be modular such that changes to modules pertaining to training (e.g., the loss function) do not impact changes to modules pertaining to inference.

#### Recommendation: Use Python for training and analysis software.

It is recommended to implement training software in Python. Python is commonly used for ML training, analysis, and visualization. Python is a *Duck-typed* language which means variables are typed according to data passed to them.  Python is an interpreted language, which means programs can run immediately without being compiled. This is convenient for programmers but can result in slower code for some tasks. Python's interpreter will compile it after first run, however, it is still primarily a scripting language, meaning it is focused on automating tasks rather than optimizing for performance by allowing direct manipulation of the computer's hardware (though some Python code can run very fast).

The following training side tools are recommended:
* `numpy` for matrix manipulation
* `scipy` for scientific computing
* `pandas` for data management
* `matplotlib` for basic plotting with a high degree of customization
* `seaborn` for more advanced plotting
* `scikit-learn` for training machine learning models (e.g., random forest, support vector machine, small neural networks)
* `opencv` for computer vision
* `pytorch` for deep learning, including reinforcement learning, convolutional neural networks, transformers, and other modern architectures
* `captum` for neural network explainability
* `lime` for model agnostic explainability
* `adversarial-robustness-toolbox` for model hardening
* `black` for Python code auto-formatting
* `flake8` for Python code linting
* `mypy` for Python code type-checking

Many other tools are possible and there are many tool choices available (e.g., PyTorch vs. TensorFlow). This list just provides a starting point.

#### Requirement: Serialize models with a secure format.

While storing model weights as pickle files is convenient for prototyping, pickle files are not secure or suitable for production systems. Proper model serialization ensures secure model storage, compatibility between projects (if there is a need to share models), and compatibility between training and inference decoupled from a particular programming language. ML research aimed at production transition must export model weights as Open Neural Network Exchange (ONNX) files to meet these goals. ONNX emerged as the standard serialization format for deep learning and has growing support for other machine learning architectures from scikit-learn. This allows exporting models to a standard format such that any project can read them, and acts as a boundary layer between training and inference.

#### Requirement: Serialize and save experiment parameters to ensure repeatability.

To ensure training pipelines are repeatable, any parameters set by the engineer performing training experiments must be serialized using a format such as YAML or JSON, and saved. This is especially important as production data pipelines and models are highly configurable with many parameters. Saving these parameters ensures the training pipeline is repeatable and provides a record of tuning performed, preventing developers from asking *"what did I set the learning rate to again?"* when trying to repeat an experiment. There are many tools for ML experiment tracking available for free and commercially, but at minimum projects must save experiment parameters so they are repeatable. This includes seed values for random number generators, hyperparameters tuned, and parameters related to pre- and post-processing employed.

#### Requirement: Serialize configuration parameters separately from model files.

If there are additional configuration parameters which are output by the training pipeline and loaded by the inference pipeline, these must be serialized separately from the model. For example, hyperparameters related to post-processing (e.g., number of model decisions to count before showing a result to the user) must be decoupled from the model in case they must be changed later on.

#### Recommendation: Use C++ for inference software (or another language which meets the requirements of the production system).

This recommendation comes with the emphasis to select a language that meets the requirements of the production system. C++ is a good default recommendation since code written in C++ can be optimized for performance (with the caveat that skilled developers must be available) and can be compiled on most systems. Some inference environments (e.g., vehicles) may come with strict timing constraints, and others (e.g., large scale simulation where many instances of a software module are invoked) might come with memory constraints. These all motivate optimizing for requirements that might be overly constraining in training time. Other requirements, such as policy, team availability, and precedents might also drive selection of inference programming language. Critically, as long as training and inference software are decoupled, the language for training can be selected with consideration for the training system requirements.

**Class 03

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 2.2. Software Hygiene

https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/software_tips/00_software_hygiene.ipynb

### 2.3. Repository Structure

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 2.4. Evaluation of New Tools

https://github.com/ruc-practical-ai/fall-2024-class-06/blob/main/notebooks/01_image_sensor_modeling_with_pybsm.ipynb

### 2.5. Collection of Examples

#### 2.5.1. Basic Python Packages
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/basic_packages_introduction/01_numpy.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/basic_packages_introduction/02_pandas.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/basic_packages_introduction/03_pandas_plots.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/basic_packages_introduction/04_matplotlib_plots.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/basic_packages_introduction/05_matplotlib_widgets.ipynb

#### 2.5.2. ML Packages
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/ml_packages_introduction/01_intro_to_scikit_regression.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/ml_packages_introduction/02_intro_to_scikit_classification.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/ml_packages_introduction/03_clustering_with_scikit.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/ml_packages_introduction/04_pytorch_introduction.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-03/blob/main/ml_packages_introduction/05_pytorch_autoencoder_example.ipynb

#### 2.5.3. PyTorch Examples
https://github.com/ruc-practical-ai/fall-2024-class-05/blob/main/notebooks/00_dataloading.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-05/blob/main/notebooks/01_torch_transforms.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-05/blob/main/notebooks/02_model_explainability_with_captum.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-05/blob/main/notebooks/03_model_saving_with_onnx.ipynb
https://github.com/ruc-practical-ai/fall-2024-class-05/blob/main/notebooks/04_visualizing_models.ipynb

## 3. ML Requirements, Best Practices, and Techniques

### 3.1. Project Planning
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.2. Selecting a Use Case and Formulating Requirements

### 3.3. Ethics Evaluation

### 3.4. Building a Team

### 3.5. Budgeting

### 3.6. Basic ML Practices

Class 04

### 3.7. Data Pipeline
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.8. Understanding Your Sensor

Class 06

https://github.com/ruc-practical-ai/fall-2024-class-06/blob/main/notebooks/01_image_sensor_modeling_with_pybsm.ipynb

https://github.com/ruc-practical-ai/fall-2024-class-06/blob/main/notebooks/02_more_pybsm_examples.ipynb

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.9. Architecture Selection
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.10. Model Training
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.11. Hyperparameter Tuning
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.12. Early Productionization

Class 07

### 3.13. Robustness and Hardening
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.14. Post-Training Scaling
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.15. Red-Teaming
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 3.16. Test and Evaluation
https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md


## 4. Research Guidance

### 4.1. Guidance on Novelty

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 4.2. Deliverables

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 4.3. Technical Communication

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 4.4. Report Structure

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

### 4.5. Guidance on Application of Generative AI in Research

https://github.com/ruc-practical-ai/fall-2024-final-project/blob/main/INSTRUCTIONS.md

https://github.com/ruc-practical-ai/syllabus

## 5. Anti-Patterns to Avoid

### 5.1. ML Anti-Patterns

### 5.2. Visualization Anti-Patterns

### 5.3. Technical Communication Anti-Patterns

### 5.4. Software Anti-Patterns

#### Anti-pattern: DO NOT Assume C++ will make Inference Faster or More Memory Efficient by Default.

The main reason many production ML inference frameworks use C++ (or another compiled language) is runtime or memory requirements. However, moving to C++ will not automatically speed up code. Any benefits of moving to a compiled language will only be realized through well-written code by developers skilled in the language. Before deciding to adopt a language for inference due to perceived performance gains, evaluate the availability of a skilled team to implement software in that language.

#### Anti-pattern: DO NOT Allow Decoupling Training and Inference to Lead to Technical Debt Accumulation in Training Software.

In many projects, offline analysis tools might not be kept up to the same standards of code review as production software. If ML training is decoupled from inference, the training software must be held to the same standards as the inference software, since it produces the weights which define the inference software execution.

### 5.5. General Common Mistakes

