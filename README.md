![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Tools prompting

## Getting Started

Follow the instructions provided in the notebook.

Read the instructions for each cell and provide your answers. Make sure to test your answers in each cell and save. Jupyter Notebook should automatically save your work progress. But it's a good idea to periodically save your work manually just in case.

## Deliverables

- Downloaded notebook with your responses to each of the exercises.

## Run the completed notebook

The solution adds three tools (temperature conversion, text reversal, and word
counting), model-driven examples, validation checks, and a reflection. It uses
plain text prompting through Ollama rather than native tool calls.

Install the Python dependencies with `python -m pip install -r requirements.txt`.
Start Ollama and download the model with `ollama pull qwen3:4b` if needed.
The model can be changed with the `OLLAMA_MODEL` environment variable.

Run and save all cell outputs:

```powershell
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=360 lab-tools-prompting.ipynb
```

Or open `lab-tools-prompting.ipynb` in Jupyter and select **Run All**.

## Submission

- Upon completion, add your deliverables to git. 
- Then commit git and push your branch to the remote.
- Make a pull request and paste the PR link in the submission field in the Student Portal.

<br>

**Good luck!**
