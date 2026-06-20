# RAG Assistant

This project includes a simple RAG path based on a local Chroma database.

The current example knowledge source is:

- `data/AcmeTech_Employee_Handbook.pdf`

The current public demo path in this repository is:

- assistant: `rag-assistant`
- vector store: local `Chroma`
- default provider path: `openai-compatible`

## Setting up Chroma

To create the local Chroma database:

1. Put your source files into `./data`. The current script supports Word and PDF files.
2. Open [`create_chroma_db.py`](../scripts/create_chroma_db.py) and confirm `folder_path` points to `./data`.
3. Adjust the database name, chunk size and overlap if needed.
4. After activating the virtual environment, run:

```sh
python scripts/create_chroma_db.py
```

5. If successful, a local Chroma directory will be created in the repository root. In the current repository version, the active demo database path is:

- `chroma_db_small_chunks/`

This version uses smaller chunks for the current handbook demo so retrieval results are less coarse than a page-level split.

## Configuring the RAG assistant

To adapt the current RAG assistant:

1. Open [`tools.py`](../src/agents/tools.py) and confirm the Chroma `persist_directory` points to the database you created.
2. Adjust the retriever document count if needed. The current default is `k=5`.
3. Keep the retrieval tool description and returned source information aligned with the real knowledge base contents.
4. Open [`rag_assistant.py`](../src/agents/rag_assistant.py) and update the assistant instructions so they match the actual knowledge domain.

```python
instructions = f"""
    You are a knowledge-base assistant focused on answering questions from a specific internal handbook.
    Today's date is {current_date}.

    IMPORTANT:
    - The user cannot see the raw tool response.
    - Only answer with information supported by the retrieved context.
    - If no relevant content is found, say so clearly.
    """
```

5. Open [`streamlit_app.py`](../src/streamlit_app.py) and update the welcome message if you change the assistant’s domain.

```python
WELCOME = """你好！我是一个知识库问答助手，可以基于内置资料回答问题。"""
```

6. Run the application and test the RAG assistant.

## Current behavior

The current implementation:

- uses Chroma as the local vector store
- uses `Knowledge_Base_Search` as the retrieval tool
- returns structured retrieval results including:
  - `hit_count`
  - `source`
  - `context`
- shows current knowledge base status on the app home screen and in the service info area

It also includes a few demo-oriented retrieval safeguards:

- fixed retrieval-first flow for `rag-assistant`
- stable no-hit response handling
- lightweight metadata for showing hit / no-hit state in the UI
- small chunk local database for the sample handbook
- lightweight English retrieval hints for Chinese handbook questions

## Recommended manual checks

After generating the Chroma database, manually verify:

1. `rag-assistant` can be selected in the UI
2. the home screen shows the current knowledge base status
3. handbook-related questions return grounded answers
4. unrelated questions clearly report that the example knowledge base did not return relevant content

## Recommended Demo Questions

Questions that should normally hit:

1. `员工手册里有没有提到远程办公政策？`
2. `员工福利相关内容主要包括什么？`
3. `员工手册里如何描述休假或请假政策？`

Questions that should normally not hit:

1. `CTO 的邮箱地址是什么？`
2. `公司今年招聘多少个实习生？`
3. `今年年终奖具体公式是什么？`

## Fast Demo Flow

If you want to quickly verify the current repository demo path:

1. Start the FastAPI service.
2. Start the Streamlit app.
3. In the UI, explicitly select:
   - assistant: `rag-assistant`
   - model: `openai-compatible`
4. Ask one hit question and one no-hit question from the list above.

The current demo should behave like this:

- hit questions return a concise answer grounded in the handbook
- the UI shows a `知识库命中` badge for hit answers
- no-hit questions clearly say the example knowledge base did not return relevant content
- the UI shows a `知识库未命中` badge for no-hit answers

## Notes For This Repository Version

This repository currently favors a small, inspectable RAG demo over a more complex architecture.

That means the current priority is:

- stable manual demo behavior
- clear hit / no-hit output
- easy local iteration in `future_repo_root/`

and not:

- multi-agent orchestration for RAG
- complex retrieval pipelines
- aggressive framework-level customization
