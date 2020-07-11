# Sequence Models Project Notes

	* [Sequence Models](#sequence-models)
	  * [Review of the whole Project and Course](#review-of-the-whole-project-and-course)
      * [Building a recurrent neural network - step by step](#building-a-recurrent-neural-network---step-by-step)
      * [Dinosaur Island -- Character-level language model](#dinosaur-island----character-level-language-model)
      * [Jazz improvisation with LSTM](#jazz-improvisation-with-lstm)
      * [Emojify](#emojify)
      * [Word Vector Representation](#word-vector-representation)
      * [Machine Translation (Neural Machine Translation)](#machine-translation-neural-machine-translation)
      * [Trigger word detection](#trigger-word-detection)
	  
This repository is the summaries of the project Sequence Models on [DeepLearning.ai](https://deeplearning.ai) specialization courses.

## Sequence Models

### Review of the whole Project and Course

> This course will teach you how to build models for natural language, audio, and other sequence data. Thanks to deep learning, sequence algorithms are working far better than just two years ago, and this is enabling numerous exciting applications in speech recognition, music synthesis, chatbots, machine translation, natural language understanding, and many others. 
>
> You will:
> - Understand how to build and train Recurrent Neural Networks (RNNs), and commonly-used variants such as GRUs and LSTMs.
> - Be able to apply sequence models to natural language problems, including text synthesis. 
> - Be able to apply sequence models to audio applications, including speech recognition and music synthesis.
>
> This is the fifth and final course of the Deep Learning Specialization.

### Building a recurrent neural network - step by step
Welcome to Course 5's first assignment! In this assignment, you will implement your first Recurrent Neural Network in numpy.

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory". They can read inputs x<sup>`<t>`</sup> (such as words) one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next. This allows a uni-directional RNN to take information from the past to process later inputs. A bidirection RNN can take context from both the past and the future. 

**Notation**:
- Superscript <sup>[l]</sup> denotes an object associated with the `l.th` layer.
    - Example: a<sup>[4]</sup> is the `4.th` layer activation. W<sup>[5]</sup> and b<sup>[5]</sup> are the `5.th` layer parameters.

- Superscript <sup>(i)</sup> denotes an object associated with the `i.th` example.
    - Example: x<sup>(i)</sup> is the `i.th` training example input.

- Superscript `<t>` denotes an object at the `t.th` time-step.
    - Example: x<sup>`<t>`</sup> is the input x at the `t.th` time-step. x<sup>`(i)<t>`</sup> is the input at the `t.th` timestep of example `i`.

- Lowerscript <sub>i</sub> denotes the `i.th` entry of a vector.
    - Example: a<sup>[l]</sup><sub>i</sub> denotes the `i.th` entry of the activations in layer `l`.

We assume that you are already familiar with `numpy` and/or have completed the previous courses of the specialization. Let's get started!

### Dinosaur Island -- Character-level language model
Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and in this assignment they are back. You are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, and your job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go beserk, so choose wisely! 
<br />
Luckily you have learned some deep learning and you will use it to save the day. Your assistant has collected a list of all the dinosaur names they could find, and compiled them into this [dataset](dinos.txt). (Feel free to take a look by clicking the previous link.) To create new dinosaur names, you will build a character level language model to generate new names. Your algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep you and your team safe from the dinosaurs' wrath! 

By completing this assignment you will learn:

- How to store text data for processing using an RNN 
- How to synthesize data, by sampling predictions at each time step and passing it to the next RNN-cell unit
- How to build a character-level text generation recurrent neural network
- Why clipping the gradients is important

We will begin by loading in some functions that we have provided for you in `rnn_utils`. Specifically, you have access to functions such as `rnn_forward` and `rnn_backward` which are equivalent to those you've implemented in the previous assignment. 

### Jazz improvisation with LSTM
Welcome to your final programming assignment of this week! In this notebook, you will implement a model that uses an LSTM to generate music. You will even be able to listen to your own music at the end of the assignment. 

**You will learn to:**
- Apply an LSTM to music generation.
- Generate your own jazz music with deep learning.

Please run the following cell to load all the packages required in this assignment. This may take a few minutes. 

### Emojify
Welcome to the second assignment of Week 2. You are going to use word vector representations to build an Emojifier. 

Have you ever wanted to make your text messages more expressive? Your emojifier app will help you do that. So rather than writing "Congratulations on the promotion! Lets get coffee and talk. Love you!" the emojifier can automatically turn this into "Congratulations on the promotion! 👍 Lets get coffee and talk. ☕️ Love you! ❤️"

You will implement a model which inputs a sentence (such as "Let's go see the baseball game tonight!") and finds the most appropriate emoji to be used with this sentence (⚾️). In many emoji interfaces, you need to remember that ❤️ is the "heart" symbol rather than the "love" symbol. But using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate words in the test set to the same emoji even if those words don't even appear in the training set. This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set. 

In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings, then build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM. 

Lets get started! Run the following cell to load the package you are going to use. 

### Word Vector Representation
Welcome to your first assignment of this week! 

Because word embeddings are very computionally expensive to train, most ML practitioners will load a pre-trained set of embeddings. 

**After this assignment you will be able to:**

- Load pre-trained word vectors, and measure similarity using cosine similarity
- Use word embeddings to solve word analogy problems such as Man is to Woman as King is to ______. 
- Modify word embeddings to reduce their gender bias 

Let's get started! Run the following cell to load the packages you will need.

### Machine Translation (Neural Machine Translation)
Welcome to your first programming assignment for this week! 

You will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). You will do this using an attention model, one of the most sophisticated sequence to sequence models. 

This notebook was produced together with NVIDIA's Deep Learning Institute. 

Let's load all the packages you will need for this assignment.

### Trigger word detection
Welcome to the final programming assignment of this specialization! 

In this week's videos, you learned about applying deep learning to speech recognition. In this assignment, you will construct a speech dataset and implement an algorithm for trigger word detection (sometimes also called keyword detection, or wakeword detection). Trigger word detection is the technology that allows devices like Amazon Alexa, Google Home, Apple Siri, and Baidu DuerOS to wake up upon hearing a certain word.  

For this exercise, our trigger word will be "Activate." Every time it hears you say "activate," it will make a "chiming" sound. By the end of this assignment, you will be able to record a clip of yourself talking, and have the algorithm trigger a chime when it detects you saying "activate." 

After completing this assignment, perhaps you can also extend it to run on your laptop so that every time you say "activate" it starts up your favorite app, or turns on a network connected lamp in your house, or triggers some other event? 

In this assignment you will learn to: 
- Structure a speech recognition project
- Synthesize and process audio recordings to create train/dev datasets
- Train a trigger word detection model and make predictions

Lets get started! Run the following cell to load the package you are going to use.  