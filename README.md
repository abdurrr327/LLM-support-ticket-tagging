# Auto-Tagging Support Tickets Using Large Language Models (LLMs)

This project demonstrates how to automatically classify and tag customer support tickets using various Large Language Model (LLM) techniques. It compares the performance of Zero-Shot Learning, Few-Shot Learning, and Fine-Tuning approaches on a real-world dataset.

## 🚀 Objective

The primary goal is to build and evaluate a system that can automatically assign one or more relevant tags (e.g., 'Technical Issue', 'Billing Inquiry') to a given support ticket based on its text content. The system should also be able to rank and provide the top 3 most likely tags for each ticket.

This helps in:
- **Automating Triage:** Automatically routing tickets to the correct department.
- **Improving Efficiency:** Reducing manual effort and response times for support agents.
- **Analytics:** Gaining insights into the most common types of customer issues.

## 📦 Dataset

The project uses the `customer_support_tickets.csv` dataset. This dataset contains fictional customer support tickets with the following key columns used for this task:

-   `Ticket Description`: The main body of the customer's support request.
-   `Ticket Subject`: The subject line of the ticket.
-   `Ticket Type`: The ground-truth category/tag for the ticket. This is our target variable for classification.

The primary task is to predict the `Ticket Type` based on the combined text from the `Ticket Subject` and `Ticket Description`.

## 🛠️ Methodology

The project is structured within a single Google Colab notebook (`Auto_Tagging_Support_Tickets.ipynb`) and explores three distinct LLM-based classification methods.

### 1. Zero-Shot Learning with GPT-4

-   **Description:** This method uses a powerful, pre-trained LLM (OpenAI's GPT-4o-mini) without any further training. We engineer a precise prompt that instructs the model on how to classify the ticket text into one of the predefined categories.
-   **Advantages:** Requires no training data or infrastructure. It is extremely fast to set up and test.
-   **Disadvantages:** Performance depends entirely on the LLM's pre-existing knowledge and the quality of the prompt. It can be more expensive per API call than a self-hosted model.

### 2. Few-Shot Learning with GPT-4

-   **Description:** This technique improves upon zero-shot learning by including a few examples of correctly classified tickets directly within the prompt. This "in-context learning" helps the model better understand the specific nuances and expected output format.
-   **Advantages:** Often provides a significant performance boost over zero-shot with minimal effort and without the need for a full fine-tuning process.
-   **Disadvantages:** Similar cost structure to zero-shot, and the prompt length increases with more examples.

### 3. Fine-Tuning with DistilBERT

-   **Description:** In this approach, we take a smaller, open-source transformer model (`distilbert-base-uncased`) and train it specifically on our support ticket dataset. The model's weights are updated to become specialized in this particular classification task.
-   **Advantages:** Can achieve very high accuracy on the specific task it was trained for. Once trained, inference is fast and can be done on local hardware, making it cost-effective at scale.
-   **Disadvantages:** Requires a labeled dataset, significant computational resources for training, and more technical expertise to implement.

## 📈 Evaluation

Each method is evaluated on a held-out test set using standard classification metrics:

-   **Accuracy:** Overall percentage of correctly classified tickets.
-   **Precision, Recall, F1-Score:** Calculated per-class to understand performance on each ticket type.
-   **Confusion Matrix:** A visualization to see where the model is making mistakes (e.g., confusing 'Billing Inquiry' with 'Refund Request').

A final summary table compares the performance of all three methods side-by-side.

## 📌 How to Run

1.  **Environment:** This project is designed to run in a Google Colab environment.
2.  **Files:**
    -   Upload the `customer_support_tickets.csv` file to your Colab session's file storage.
    -   Open the `Auto_Tagging_Support_Tickets.ipynb` notebook in Google Colab.
3.  **API Key:**
    -   For the Zero-Shot and Few-Shot sections, you will need an **OpenAI API Key**.
    -   Insert your key in the designated code cell: `os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"`.
4.  **Execution:** Run all cells in the notebook from top to bottom. The notebook is structured with clear headings and markdown cells explaining each step.

## Results & Insights

-   **Zero-Shot Learning** provided a decent baseline, demonstrating the impressive general capabilities of modern LLMs.
-   **Few-Shot Learning** significantly improved accuracy over the zero-shot approach, showing the power of in-context examples for guiding the model.
-   **Fine-Tuning DistilBERT** achieved the highest performance on all metrics. This is expected, as the model was specifically trained on the task's data distribution and nuances.

**Conclusion:** For quick prototyping or tasks with limited data, Few-Shot Learning with a powerful API is highly effective. For production systems where high accuracy and low inference cost are critical, fine-tuning a smaller, dedicated model is the superior approach.
