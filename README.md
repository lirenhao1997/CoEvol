# CoEvol
CoEvol: Constructing Better Responses for Instruction Finetuning through Multi-Agent Cooperation
## News
- ⭐ [09/2024] The paper has been accepted to the main conference of EMNLP 2024.
- 💻 [06/2024] The main code for CoEvol has been released.
- 📰 [06/2024] The paper has been released on [Arxiv](https://arxiv.org/abs/2406.07054).
- 🔥 [06/2024] The evolved datasets and fine-tuned models powered by CoEvol have been released in this [huggingface collection](https://huggingface.co/collections/CAS-SIAT-ConsistencyAI/coevol-66683b34d45cc54b889c532d).

## How to run CoEvol?
### Installation
```bash
  git clone https://github.com/lirenhao1997/CoEvol.git
  cd CoEvol
  pip install -r requirements.txt
```

### Data Preparation
- Theoretically, CoEvol is capable of refining any IFT data sample that includes an instruction and an original response.
In our implementation, data formats aligned to Alpaca and ShareGPT are supported.
- In our paper, we explored the potential of CoEvol on top of high-quality data after the data selection process employed by [Deita](https://github.com/hkust-nlp/deita). Please refer to their repository for a detailed data selection pipeline.
- The formatted data should be placed in the ```data/``` directory, in alignment with the ```--root_path <YOUR_DATA_PATH>``` and ```--dataset_name <YOUR_DATASET_NAME>``` specification in the running scripts.

### Environment Configuration

**Run CoEvol via External API**

CoEvol can run on proprietary models via APIs. Until now, we have tested our framework on the APIs of OpenAI, GLM, ERNIE, and custom proxies. To run CoEvol via external APIs, you should:
1. Set your API keys in the appropriate fields within ```edit/api_keys.json```.
2. Run the example scripts located in the directory ```scripts/```. If you want to run CoEvol based on a custom proxy, remember to set the URL with ```--proxy_api_url <YOUR_CUSTOM_PROXY>```.

**Run CoEvol via Local Deployment**

CoEvol can also run on open-source models via local deployment. To facilitate faster agent interactions, we highly recommend utilizing inference acceleration techniques. In this implementation, we utilize [vllm](https://github.com/vllm-project/vllm) for local inference acceleration, which includes a chat API that is compatible with OpenAI services.
1. Deploy your local model with vllm. For more detailed settings, please refer to the [official documents](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html).
```
python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model "<YOUR_MODEL_PATH>" \
	--served-model-name "<YOUR_MOEL_ALIAS>" \
    --tensor-parallel-size 4
```
2. Run the example script in ```run_iter_pipeline_local.sh``` depending on whether you use single or multi-turn data.

**Notes**
1. Use the parameters ```--save_mem``` and ```--save_log``` to save agent memories and running logs, respectively.
2. Employ the parameters ```--start_indx``` and ```--end_indx``` to control the range of data evolution. If these parameters are not set, CoEvol will process the entire dataset for data evolution.
3. Utilize the parameter ```--num_workers``` to control the number of multi-threads used for concurrent data evolution, which should be adjusted to be compatible with the rate limit of your APIs or the load capacity of your local server.

### Data Organization for SFT
Once you successfully run the framework, both intermediate processes and full results will be stored in the directory ```./edit/res/<JOB_NAME>```.
To obtain the evolved SFT data in JSON format, use the appropriate functions within the script ```edit/data_post_process.py```, according to the data you have used.

## Fine-tuning with Evolved Data
For supervised fine-tuning, we utilize [llama-factory](https://github.com/hiyouga/LLaMA-Factory) to train our model. Please consult their repository for detailed instructions.

## Citation
If you find the content of this project helpful, please cite our paper as follows:

```
@misc{li2024coevol,
      title={CoEvol: Constructing Better Responses for Instruction Finetuning through Multi-Agent Cooperation}, 
      author={Renhao Li and Minghuan Tan and Derek F. Wong and Min Yang},
      year={2024},
      eprint={2406.07054},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement
For conversation prompt templates, we use codes from [fastchat](https://github.com/lm-sys/FastChat).
