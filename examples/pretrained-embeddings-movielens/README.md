# Training Recommender Systems with Pretrained Embeddings

This project demonstrates how to build a recommender system with pretrained embeddings.

Pretrained embeddings can come from many sources. They can be a side product of other models trained on organizational data. They might come from models trained specifically with obtaining embeddings in mind, such as product2vec. Or, as will be the case in this project, they can represent data of other modalities (images, in our case) to our model. 

We will use the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) and enrich it with image data (movie posters downloaded from IMDB). We will then leverage a pretrained CNN, namely ResNet-50, to extract image features. Finally, we will feed the multi-modal data to our network for the task of predicting user-movie rating scores.


## Docker image

The notebooks in this tutorial are guaranteed to work with the `merlin-tensorflow-training:22.04` image. Please find the instructions for running the container below.

```
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.05
docker run --gpus=all -it --rm --net=host --ipc=host -v ${PWD}:/workspace nvcr.io/nvidia/merlin/merlin-tensorflow-training:22.05
```

Then, from within the container, start Jupyter Notebook:

```
jupyter notebook --allow-root --no-browser --NotebookApp.token='' --ip='0.0.0.0'
```

The notebooks should be executed in the following order.

- [01-Download-Convert.ipynb](01-Download-Convert.ipynb)

- [02-Data-Enrichment.ipynb](02-Data-Enrichment.ipynb)

- [03-Feature-Extraction-Poster.ipynb](03-Feature-Extraction-Poster.ipynb): this notebook is executed using a ResNet container. See details in the notebook.

- [04-Feature-Extraction-Text.ipynb](04-Feature-Extraction-Text.ipynb): this notebook is executed using a HuggingFace NLP container. See details in the notebook.

- [05-Create-Feature-Store.ipynb](05-Create-Feature-Store.ipynb)

- [06a-Training-with-TF-with-pretrained-embeddings.ipynb](06a-Training-with-TF-with-pretrained-embeddings.ipynb)

- [06b-Training-wide-and-deep-with-pretrained-embedding.ipynb](06b-Training-wide-and-deep-with-pretrained-embedding.ipynb)
