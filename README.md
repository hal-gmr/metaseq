# Fine Tuning Project - Hal Johnson
Fine tuning the massive OPT 175 to recognize similar names and persons by their human names.
This project improves its performance and efficiency by focusing on a specific area of expertise.

#Prerequisites
Meta's 17b billion param ai model aka OPT-175B
A dataset for the specialized names use case
A suitable computing environment (e.g., GPU-enabled machine or cloud instance)
#Step 1: Prepare the Dataset
Collect and preprocess a dataset suitable for fine-tuning (e.g., JSON)
Split the dataset into training, validation, and testing sets (e.g., 80% for training, 10% for validation, and 10% for testing)
#Step 2: Set Up the Fine-Tuning Environment
Install PyTorch and the Hugging Face Transformers library
Clone the Meta OS LLM repository and navigate to the directory
Install the required dependencies and setup the environment
#Step 3: Load the Pretrained Model
Import the OS LLM model and load the pretrained weights
Configure the model for fine-tuning (e.g., set the number of layers to freeze)
#Step 4: Fine-Tune the Model
Create a custom dataset class for the name dataset
Define a data loader for the training and validation sets
Set up the fine-tuning loop, including loss calculation, backward pass, and optimization
Monitor performance on the validation set and adjust hyperparameters as needed
#Step 5: Evaluate and Test the Fine-Tuned Model
Evaluate the fine-tuned model on the test set
Compare performance to the original OS LLM and other baselines
Test the model on new, unseen data to ensure generalizability
#Step 6: Deploy the Fine-Tuned Model
Save the fine-tuned model weights and configuration
Deploy the model in an application or service
Monitor performance and retrain as necessary


# Metaseq
A codebase for working with [Open Pre-trained Transformers](projects/OPT), originally forked from [fairseq](https://github.com/facebookresearch/fairseq).


## Community Integrations

### Using OPT with 🤗 Transformers

The OPT 125M--66B models are now available in [Hugging Face Transformers](https://github.com/huggingface/transformers/releases/tag/v4.19.0). You can access them under the `facebook` organization on the [Hugging Face Hub](https://huggingface.co/facebook)

### Using OPT-175B with Alpa

The OPT 125M--175B models are now supported in the [Alpa project](https://alpa-projects.github.io/tutorials/opt_serving.html), which 
enables serving OPT-175B with more flexible parallelisms on older generations of GPUs, such as 40GB A100, V100, T4, M60, etc.

### Using OPT with Colossal-AI

The OPT models are now supported in the [Colossal-AI](https://github.com/hpcaitech/ColossalAI#OPT), which helps users to efficiently and quickly deploy OPT models training and inference, reducing large AI model budgets and scaling down the labor cost of learning and deployment.

### Using OPT with CTranslate2

The OPT 125M--66B models can be executed with [CTranslate2](https://github.com/OpenNMT/CTranslate2/), which is a fast inference engine for Transformer models. The project integrates the [SmoothQuant](https://github.com/mit-han-lab/smoothquant) technique to allow 8-bit quantization of OPT models. See the [usage example](https://opennmt.net/CTranslate2/guides/transformers.html#opt) to get started.

### Using OPT with FasterTransformer

The OPT models can be served with [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), a highly optimized inference framework written and maintained by NVIDIA. We provide instructions to convert OPT checkpoints into FasterTransformer format and [a usage example](docs/faster-transformer.md) with some benchmark results.

### Using OPT with DeepSpeed

The OPT models can be finetuned using [DeepSpeed](https://github.com/microsoft/DeepSpeed). See the [DeepSpeed-Chat example](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) to get started.

## Getting Started in Metaseq
Follow [setup instructions here](docs/setup.md) to get started.

### Documentation on workflows
* [Training](docs/training.md)
* [API](docs/api.md)

### Background Info
* [Background & relationship to fairseq](docs/history.md)
* [Chronicles of training OPT-175B](projects/OPT/chronicles/README.md)

## Support
If you have any questions, bug reports, or feature requests regarding either the codebase or the models released in the projects section, please don't hesitate to post on our [Github Issues page](https://github.com/facebookresearch/metaseq/issues).

Please remember to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributing
We welcome PRs from the community!

You can find information about contributing to metaseq in our [Contributing](docs/CONTRIBUTING.md) document.

## The Team
Metaseq is currently maintained by the CODEOWNERS: [Susan Zhang](https://github.com/suchenzang), [Naman Goyal](https://github.com/ngoyal2707), [Punit Singh Koura](https://github.com/punitkoura), [Moya Chen](https://github.com/moyapchen), [Kurt Shuster](https://github.com/klshuster), [David Esiobu](https://github.com/davides), [Igor Molybog](https://github.com/igormolybogFB), [Peter Albert](https://github.com/Xirider), [Andrew Poulton](https://github.com/andrewPoulton), [Nikolay Bashlykov](https://github.com/bashnick), [Binh Tang](https://github.com/tangbinh), [Uriel Singer](https://github.com/urielsinger), [Yuchen Zhang](https://github.com/zycalice), [Armen Aghajanya](https://github.com/ArmenAg), [Lili Yu](https://github.com/lilisierrayu), and [Adam Polyak](https://github.com/adampolyak).

## License

The majority of metaseq is licensed under the MIT license, however portions of the project are available under separate license terms: 
* Megatron-LM is licensed under the [Megatron-LM license](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE)

