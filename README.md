# TabularGPT

The idea is similar to what is described in these papers:
1. [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959v3.pdf)
2. [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678v1.pdf)
3. [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)

While closest to 1., it has some slight changes, and several potential areas for improvement

## Areas for improvement

### 1. Using this idea for Multimodality

Essentially, this proves that the transformer architecture, specifically multi-head self attention applied to tokens/embeddings, can work not only for NLP (words), not only for CV (images), but now also near state-of-the-art for tabular data. The idea of embeddings is clearly generalizable, so as long as the various modalities can feed into the transformer as "words" (as we originally though of transformers for NLP), then we can get high signal to noise ratio under this clean generalizable architecture

In NLP, the transformer might handle a sentences of length say 1024 where most of it is usually padding at the end. In TabularGPT, our "sentences" are fixed to be the number of columns, so in the Iris dataset it would always be a 4-length sentence with SepalLength, SepalWidth, PetalLength, PedalWidth. In the MultimodalGPT, our sentences would be the number of embeddings we want to use, however we suppose attention applied to the embeddings may generate good signal. For example, we might have some tabular columns occupy say 4 embeddings (lets use Iris again), but also we might have say 9 more embeddings for an image chopped up into 9 squares (like in the ViT paper). So in this example, we would have 13 embeddings, and the model would try to find how the output relates to the 13 embeddings/"words"

### 2. TabularGPT's version of "ViViT"

How might the tabular data, say a patient's age or blood pressure interact with what the model is seeing? That is the goal. Or, you can still take the output of the ViT, call that an embedding, and see how that embedding interacts with the tabular data. This sort of leads to the generalization, which is akin to the ViViT, where you pack embeddings from the transformer into another transformer!

As mentioned above, the idea that we can take embeddings from anywhere, including another transformer (or CNN etc), and plug them into another transformer is essentially the idea behind ViViT, which takes the embeddings from the images an applies it to the video

### 3. A pretraining step -- or some other way to optimize embeddings

A key of this will be the richness of the embeddings. For tabular data, do we just want to project it up? Do we want to use bias if we do so (Paper #1 suggest to do)? Maybe do we want to add an activation for some nonlinearity? Is it better to use nn.Embedding for categorical data than to onehot encode like Paper #1? If so is a nonlinear embedding good or do we want a linear one -- or does it depend on the problem? Lot of question to be answered, but we certainly got something here!

## Areas for Concern

### 1. Embeddings for tabular data are meaningless

It can be just simply a projection up from 1-dim to n-dim, but we hope the dimensions will have some impact/meaning eventually. While it starts out as a random projection, perhaps some dimensions eventually carry some useful signal as backprop happens.

I also worry they might be close in n-dim space with other similar embedding/vectors that don't actually have any relation to it, and so the model thinks that those embeddings, because they are close in n-dim space, are similar, but it was just because random initialization or something. This is why I think a pre-train step to predict hidden columns or something might be crucial to find "meaningful embeddings" for tabular data -- especially if we then want to combine them with say rich image or word embeddings!

### 2. Do we want to use positional encodings? 

That would allow the embeddings to have a way of aligning themselves with the column they represent, but does it matter?

Yes, it could matter because ex: say we project up to 16 dimensions, dim-2 for col-1 has no relation to dim-2 for col-2, so adding positional encodings means we can add some meaningful differentiation between the two

No, it should not matter because the model should find patterns useful to predictions in the dimensions as it trains

*Needs Experimentation...*
