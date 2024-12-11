# Basic transformer Trainer for Matmul-Free/limited with QAT
Transformer Trainer for Matmul-free or limited implementation w/ inference

Hello! This is a training (and inference) program BASED on the Matmul-Free architecture outlined in 
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

This implementation is extremely basic, with minimal logging and the most direct impleentation that I found that works. This currently saves the file as QAT for continued training, it must be converted with "torch.quantization.convert(model, inplace=True)" to use it outside of this (the accompanied inference trainer ignores these keys so you don't need to convert it to test the inference, although I'm not certain it will be accurate to its values after conversion.) 

I should note, this is not a perfect recreation as I lack the necessary hardware to fully implement this system, as it requires a large amount of info to be held on hardware which while easy for inference, creates limitations during training. It is possible to overcome this limitation with smart caching and clearing, but I just wanted something that worked. I haven't yet fully completed a transformer, but based on the loss calculations everything APPEARS to be working correctly.

I should note, there are some small bugs due to this being unfinished and numerous revisions during this process. At one point it was abled to use both chunked and unchunked datasets of any type, which I will also upload, but I was not satisfied with the implementation of the architecture. 

As it is now, the traning data can be in multiple formats with certain restrictions. .txt files are fine, a parquet or json with text, TEXT, messages will work, as well as one with instruct and output columns which will be concatenated. Or a csv with text columns or instruct/output columns.

To load a dataset, have the files you want to use alone in a folder, and press "select dataset directory" and select the folder. Press "load dataset" to load it. Then, press "select/create tokenized data" and choose no when asked to use existing tokenized data. Enter a new name for a file. Then, you can press "tokenize data" to laod it (after you laod a tokenizer.) If you use a pre-tokenized file you must still press tokenzie daa to load it.

Then you can press start training after adjusting your parameters and loading your model and tokenizer. Note: when pressing stop training, it will have a second dialog box pop up when it actually stops after the current batch is completed so you can save a model if you stop mid training. 

You can also create a tokenizer from a vocab.json. 

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
