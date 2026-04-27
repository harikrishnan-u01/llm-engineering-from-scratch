"""
Reusable tool functions for agent scripts.
Each function takes a string input and returns a string result.
"""
import ast
import operator
import datetime


def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression. Input: a math expression like '15 * 47.50 / 100'."""
    import math

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }
    allowed_funcs = {
        "round": round,
        "abs": abs,
        "int": int,
        "float": float,
        "sqrt": math.sqrt,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(_eval(node.operand))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                raise ValueError(f"Unsupported function: {ast.unparse(node.func)}")
            return allowed_funcs[node.func.id](*[_eval(a) for a in node.args])
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    try:
        tree = ast.parse(expression.strip().strip("'\"").replace("$", ""), mode="eval")
        result = _eval(tree.body)
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def get_date(query: str = "") -> str:
    """Return today's date with weekday number for calendar arithmetic. Input is ignored."""
    today = datetime.date.today()
    iso = today.isoweekday()  # 1=Monday, 2=Tuesday, ..., 7=Sunday
    days_until_monday = (8 - iso) % 7 or 7  # always 1-7; today=Monday → 7
    return (
        f"Today's date is {today.strftime('%A, %B %d, %Y')}. "
        f"Weekday number: {iso} (1=Monday, 7=Sunday). "
        f"Days until next Monday: {days_until_monday}."
    )


def search_docs(query: str, chroma_path: str = "./data/chroma_db") -> str:
    """Search the local ChromaDB vector store and return top relevant chunks."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        if not collections:
            return "No documents indexed yet. Run 06_rag_langchain.py first to build the index."

        collection = client.get_collection(
            name=collections[0].name,
            embedding_function=embedding_functions.OllamaEmbeddingFunction(
                url="http://localhost:11434/api/embeddings",
                model_name="nomic-embed-text",
            ),
        )
        results = collection.query(query_texts=[query], n_results=3)
        chunks = results["documents"][0] if results["documents"] else []
        if not chunks:
            return "No relevant content found."
        return "\n---\n".join(chunks)
    except Exception as e:
        return f"Search error: {e}"
