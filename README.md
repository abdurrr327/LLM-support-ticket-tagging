Auto-Tagging Support Tickets Using LLMs
This repository contains a Google Colab notebook demonstrating how to use Large Language Models (LLMs) to automatically categorize customer support tickets. The project explores and compares three popular techniques: Zero-Shot Learning, Few-Shot Learning, and Fine-Tuning.
Objective
The primary goal of this project is to build an automated system that can accurately classify incoming support tickets into predefined categories. This system aims to:
Reduce manual effort and time spent on ticket triage.
Improve the efficiency of customer support workflows by routing tickets to the correct teams automatically.
Evaluate and compare the performance of different LLM-based classification methods.
Provide not just a single prediction, but a ranked list of the top 3 most likely categories for each ticket, allowing for human-in-the-loop verification.
Dataset
The project uses the customer_support_tickets.csv dataset. This dataset contains fictional customer support ticket data, including a Ticket Description (the main text) and a Ticket Type (the category label).
Columns of interest:
Ticket Description: The body of the support ticket.
Ticket Subject: The subject line of the ticket.
Ticket Type: The ground truth category for the ticket (e.g., "Technical issue", "Billing inquiry").
Methodology / Approach
The core of the project is to treat the Ticket Description and Ticket Subject as input text and predict the Ticket Type. We implemented and evaluated three distinct LLM-based approaches.
1. Data Loading & Preprocessing
The dataset was loaded using the Pandas library.
The Ticket Subject and Ticket Description columns were concatenated to create a single, comprehensive text column for model input.
The target labels were extracted from the Ticket Type column.
The dataset was split into training and testing sets for the fine-tuning and evaluation stages.
2. Zero-Shot Classification
Concept: This method uses a model pre-trained on a Natural Language Inference (NLI) task to classify text into categories it has never seen during training. The model determines the probability of a "premise" (the ticket text) entailing a "hypothesis" (the category label).
Model Used: facebook/bart-large-mnli
Implementation: We used the Hugging Face pipeline for a straightforward implementation. We passed the ticket text and the list of all possible categories to the model.
3. Few-Shot Learning (In-Context Learning)
Concept: This technique leverages a large, instruction-tuned generative model. We provide the model with a carefully crafted prompt that includes instructions, the list of possible categories, and a few examples ("shots") of correctly classified tickets. The model then uses this context to classify a new, unseen ticket.
Model Used: google/flan-t5-large
Implementation: We designed a prompt template that included 3-5 examples. A function was created to dynamically insert a new ticket into this template and generate a prediction. This is a form of prompt engineering, not model training.
4. Fine-Tuning
Concept: This approach involves taking a pre-trained transformer model and further training it on our specific support ticket dataset. This adapts the model's internal weights to become an expert at our particular classification task.
Model Used: distilbert-base-uncased (a smaller, faster version of BERT).
Implementation:
The dataset was tokenized and prepared using the Hugging Face datasets library.
The AutoModelForSequenceClassification class was used, configured for our specific number of labels.
The model was trained for 3 epochs using the Trainer API, which handles the training loop, optimization, and evaluation.
A key feature of this approach is its ability to output logits for each class, which can be converted to probabilities using a Softmax function. This allows us to easily rank the predictions and find the Top 3 most probable tags for any given ticket.
Key Results & Observations
The performance of each model was evaluated on a held-out test set using standard classification metrics.
Method	Model	Accuracy	F1-Score (Macro)
Zero-Shot	facebook/bart-large-mnli	82.2%	0.77
Few-Shot	google/flan-t5-large	74.4%	0.69
Fine-Tuned	distilbert-base-uncased	95.6%	0.95
Fine-Tuning is Superior: As expected, the fine-tuned model significantly outperformed the other methods, achieving over 95% accuracy. This demonstrates the power of adapting a pre-trained model to a specific domain and task.
Zero-Shot is a Strong Baseline: The zero-shot approach provided a surprisingly strong baseline with over 82% accuracy, requiring no training data or setup time. This makes it an excellent choice for quick prototyping or when labeled data is scarce.
Few-Shot Performance: The few-shot model's performance was lower than the zero-shot model in this case. This can be attributed to several factors: the model's inherent generative nature (which can be less precise for classification than a dedicated NLI model), the sensitivity to the prompt's structure, and the limited number of examples provided in the context. More sophisticated prompt engineering could likely improve these results.
Top 3 Predictions
The fine-tuned model can provide a ranked list of predictions, which is incredibly useful for a semi-automated system.
Example Input Ticket:
"My {product_purchased} is making strange noises and not functioning properly. I suspect there might be a hardware issue. Can you please help me with this?"
Model's Top 3 Predictions:
Technical issue (Probability: 98.7%)
Product inquiry (Probability: 0.8%)
Refund request (Probability: 0.3%)
This output provides a high-confidence primary tag and reasonable secondary options, allowing a human agent to quickly verify or re-assign the ticket if needed.
Conclusion
For building a robust, production-level support ticket classification system, fine-tuning a transformer model like DistilBERT is the recommended approach, provided a sufficiently large labeled dataset is available. For rapid prototyping or situations with no labeled data, zero-shot classification offers an excellent and immediate baseline.
