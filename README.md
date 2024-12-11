# Revised transformer Trainer for Matmul-Free/limited
Transformer Trainer for Matmul-free or limited implementation, with specialized dataset preparation and multi-head attention

Hello! This is a modified training (and inference) program BASED on the Matmul-Free architecture outlined in 
Scalable MatMul-free Language Modeling
by Rui-Jie Zhu1
, Yu Zhang2
, Ethan Sifferman1
, Tyler Sheaves3
, Yiqiao Wang4
, Dustin Richmond1
, Peng Zhou1,4
, Jason K. Eshraghian1∗
1University of California, Santa Cruz 2Soochow University
3University of California, Davis 4LuxiTech

This version includes prototype behavior for multi-head attention and mirr-neuron based training behavior, breaking down datasets into imitation, completion, or response (standard) query-target pairs, to approximate the behavior of mirror neuron learned behavior in humans. Currently has a Bitnet based linear activation, not exactly like the one in the paper, trying to figure it out faces numerous issues even in this implementation.

I should note, this is not a perfect recreation as I lack the necessary hardware to fully implement this system, as it requires a large amount of info to be held on hardware which while easy for inference, creates limitations during training. It is possible to overcome this limitation with smart caching and clearing, but I just wanted something that worked. I haven't yet fully completed a transformer, but based on the loss calculations everything APPEARS to be working correctly.

I should note, there are some small bugs due to this being unfinished and numerous revisions during this process. At one point it was abled to use both chunked and unchunked datasets of any type, which I will also upload, but I was not satisfied with the implementation of the architecture. 

As it is now, the traning data must be in query-target pairs .JSON of this format (like in the smoltalk dataset):

[
    {
        "content": "Hey!",
        "role": "user"
    },
    {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
    },
etc.
}

I will include the parquet to json converter i used to convert them, by selecting only the "messages" column.

To load a dataset, leave "chunked dataset" unchecked, have the .json or .jsons you want to use alone in a folder, and press "select dataset directory" and select the folder. Press "load dataset" to load it. Then, check the "use chunked datset" checkbox, and press "select/create tokenized data" and choose no when asked to use existing tokenized data. Create a new folder for the chunked dataset files, and select that folder. Then, you have to press "load dataset" and click on the folder with the chunked file. Then you can press start training after adjusting your parameters and loading your model and tokenizer. Note: when pressing stop training, it will have a second dialog box pop up when it actually stops after the current batch is completed so you can save a model if you stop mid training. 

You can create a tokenizer from a vocab.json as well.

**I tried to fix some of the workflow behavior, so it should follow an easier path of covering for scenarios such as a folder or file not being creatd or selected.***

I hope to add on to this and improve it over time. 

Citations:

@misc{zhu2024scalablematmulfreelanguagemodeling,
      title={Scalable MatMul-free Language Modeling}, 
      author={Rui-Jie Zhu and Yu Zhang and Ethan Sifferman and Tyler Sheaves and Yiqiao Wang and Dustin Richmond and Peng Zhou and Jason K. Eshraghian},
      year={2024},
      eprint={2406.02528},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.02528}, 
}

@misc{allal2024SmolLM2,
      title={SmolLM2 - with great data, comes great performance}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Lewis Tunstall and Agustín Piqueres and Andres Marafioti and Cyril Zakka and Leandro von Werra and Thomas Wolf},
      year={2024},
}
@misc{wang2023bitnetscaling1bittransformers,
      title={BitNet: Scaling 1-bit Transformers for Large Language Models}, 
      author={Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
      year={2023},
      eprint={2310.11453},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.11453}, 
}
