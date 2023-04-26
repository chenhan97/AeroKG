# AeroKG
We introduce an automatic knowledge graph construction (KGC) tookit in aerospace/engineering domain. The KGC tookit is a pipeline model that integrates Named Entity Recognition (NER), Relation Extraction (RE), Coreference Resolution (CR), and Knowledge Graph Alignment (KGA).   
## Features
- RE model is a fine-tuned pre-trained language model (PLM), BERT, based on distantly supervised engineering domain relation extraction datasets.
- KGA model is a unsupervised entity/relation matching model that reduces duplication of extracted entities/relation by aligning them to KG.
- CR model is based on NeuralCoref 4.0 in SpaCY.
- NER model is pre-trained from [bluenqm]https://huggingface.co/bluenqm/AerospaceNER
## Installation
The detailed requirements will be shown in requirement.txt
## Usage
To run the pipeline, please use the following code:

```python
import pipeline
kg = pipeline("path_to_text")
kg.construct()
kg.visulize() #draw the knowledge graph
```

## Future Updates
- We will label and release the engineering domain datasets based on distant supervision (DS).
- We will fine-tune the relation extraction model based on the DS-labeled datasets.
- We will modify the CR module so that it can be integrated with current NRE and RE
- (optional) Package with pip
- (optional) A detailed documentation
